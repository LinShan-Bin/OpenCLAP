# Copyright 2025 CLAP Team. All rights reserved.
# Licensed under the MIT License.
"""Synchronous websocket inference server for QwenPIKM on Astribot S1.

This is the **sync** counterpart to ``async_policy_server.py``. The protocol
is the simple request/response one shared by ``deployment/model_server``:

    1. client connects → server sends ``metadata`` (msgpack-numpy)
    2. for each step:
         client → server   ``{"images": list[ndarray], "instruction": str,
                              "state": ndarray[34], "robot_type": str}``
         server → client   ``{"actions": ndarray[chunk, 34],
                              "obs_timestamp": float}``

Compared to the dual-channel async server:

* No multiprocessing, no shared memory — the model runs in the same event
  loop as the websocket handler. Useful when control loop frequency is low
  enough that synchronous round-trips fit in the budget.
* No reset frame — clients reset by closing & reopening the connection.

Usage
-----
::

    bash examples/Astribot/eval_files/run_sync_policy_server.sh \
        CKPT=/path/to/checkpoint PORT=8000

Pair with :mod:`examples.Astribot.eval_files.sync_policy_client`.
"""

from __future__ import annotations

import argparse
import logging
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from examples.Astribot.astribot_transforms import (
    LEFT_ARM_SLICE,
    LEFT_GRIPPER,
    RIGHT_ARM_SLICE,
    RIGHT_GRIPPER,
    AstribotStats,
    load_stats_for_robot,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QwenPIKMSyncPolicy
# ---------------------------------------------------------------------------
class QwenPIKMSyncPolicy:
    """Synchronous policy adapter for Astribot S1.

    Wraps the generic :class:`PolicyServerWrapper` and converts the model's
    14-dim normalized delta dual-arm output back to the 34-dim absolute
    robot command layout the Astribot client expects::

        [torso(9), left_arm_so3(9), left_grip(1),
         right_arm_so3(9), right_grip(1),
         head(2), chassis(3)]

    Head and chassis (dims 29..34) are filled from the most recent state
    snapshot — the policy doesn't predict them.
    """

    def __init__(
        self,
        ckpt_path: str,
        stats_path: str,
        robot_type: str = "S1-stationary",
        default_prompt: str = "",
        image_height: int = 240,
        image_width: int = 320,
        train_freq: int = 30,
        use_bf16: bool = True,
        device: str = "cuda",
    ) -> None:
        self._wrapper = PolicyServerWrapper(
            ckpt_path=ckpt_path,
            device=device,
            use_bf16=use_bf16,
        )
        self._robot_type = robot_type
        self._default_prompt = default_prompt
        self._image_size = (image_height, image_width)
        self._train_freq = train_freq
        self._stats: AstribotStats = load_stats_for_robot(stats_path, robot_type)
        logger.info(
            "QwenPIKMSyncPolicy ready: ckpt=%s robot=%s prompt=%r",
            ckpt_path,
            robot_type,
            default_prompt,
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = dict(self._wrapper.metadata)
        meta.update(
            {
                "robot_type": self._robot_type,
                "image_size": self._image_size,
                "train_freq": self._train_freq,
                "action_layout": (
                    "[l_xyz(3), l_eul(3), l_grip(1), "
                    "r_xyz(3), r_eul(3), r_grip(1)] absolute (xyz_rpy)"
                ),
                "action_dim": 14,
            }
        )
        return meta

    # ------------------------------------------------------------------
    # Wrapper expected by WebsocketPolicyServer: a callable accepting one
    # query dict and returning one response dict.
    # ------------------------------------------------------------------
    def __call__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return self.infer(query)

    def infer(self, query: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        # Mirror the obs key layout used at training time
        # (clap/data_transform.py:KeyMapping). We accept either the new flat
        # layout used by OpenCLAP servers (``images``/``state``) or the
        # ``images_dict.head.rgb`` / ``cartesian_so3_dict.cartesian_pose_state``
        # layout used by the upstream deploy script — pick whichever is
        # present.
        if "images" in query:
            images = query["images"]
        else:
            images = [query[k] for k in (
                "images_dict.head.rgb",
                "images_dict.left.rgb",
                "images_dict.right.rgb",
            ) if k in query]
        instruction = (query.get("instruction") or query.get("task")
                       or self._default_prompt)

        if "cartesian_so3_dict.cartesian_pose_state" in query:
            state34 = np.asarray(query["cartesian_so3_dict.cartesian_pose_state"],
                                 dtype=np.float32)
        else:
            state34 = np.asarray(query["state"], dtype=np.float32)
        if state34.shape != (34,):
            raise ValueError(f"Expected state shape (34,), got {state34.shape}")

        pil_images = [self._to_pil(im) for im in images]

        examples = self._build_example(pil_images, instruction, state34)
        out = self._wrapper.infer(examples)                     # dict, "actions": (T, 14)
        actions14 = np.asarray(out["actions"], dtype=np.float32)
        if actions14.ndim == 3:
            actions14 = actions14[0]

        # Return an absolute xyz_rpy chunk per arm (cumulative deltas applied
        # against the current state). The Astribot client converts Euler→quat
        # before calling ``move_cartesian_waypoints``.
        actions14_abs = self._adapt_actions(actions14, state34)

        return {
            "actions": actions14_abs,
            "obs_timestamp": float(time.time()),
            "infer_seconds": time.time() - t0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_pil(self, im: np.ndarray) -> Image.Image:
        if isinstance(im, Image.Image):
            return im.resize((self._image_size[1], self._image_size[0]))
        arr = np.asarray(im)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return Image.fromarray(arr).resize((self._image_size[1], self._image_size[0]))

    def _build_example(
        self,
        pil_images: List[Image.Image],
        instruction: str,
        state34: np.ndarray,
    ) -> Dict[str, Any]:
        # Convert 34-dim absolute state to the 14-dim per-arm xyz-rpy + grip
        # the policy expects, normalized via per-robot quantile bounds.
        from examples.Astribot.astribot_transforms import _so3_to_xyz_euler_np  # type: ignore

        state14 = np.zeros(14, dtype=np.float32)
        state14[0:6] = _so3_to_xyz_euler_np(state34[LEFT_ARM_SLICE])
        state14[6] = state34[LEFT_GRIPPER] / 100.0
        state14[7:13] = _so3_to_xyz_euler_np(state34[RIGHT_ARM_SLICE])
        state14[13] = state34[RIGHT_GRIPPER] / 100.0

        q01, q99 = self._stats.state_q01, self._stats.state_q99
        rng = np.maximum(q99 - q01, 1e-8)
        state14_norm = 2.0 * (state14 - q01) / rng - 1.0
        state14_norm = np.clip(state14_norm, -1.0, 1.0)

        return {
            "images": pil_images,
            "instruction": instruction,
            "state": state14_norm.astype(np.float32),
            "robot_type": self._robot_type,
        }

    def _adapt_actions(self, actions14: np.ndarray, state34: np.ndarray) -> np.ndarray:
        """Un-normalize the policy's 14-dim delta output and roll it into an
        absolute per-arm xyz_rpy chunk.

        Output layout per row is the same as the policy's training space:
          ``[l_xyz(3), l_eul(3), l_grip(1), r_xyz(3), r_eul(3), r_grip(1)]``
        Position is cumulative-summed, Euler is added per step (small-angle
        approximation — fine because train-time deltas are tiny per frame),
        gripper is rescaled from [-1, 1] to [0, 100]. The Astribot client
        converts Euler→quat before calling the SDK.
        """
        from examples.Astribot.astribot_transforms import _so3_to_xyz_euler_np  # type: ignore

        T = actions14.shape[0]
        q01, q99 = self._stats.action_q01, self._stats.action_q99
        rng = np.maximum(q99 - q01, 1e-8)

        # Un-normalize back to delta-action space [-1,1] -> raw deltas
        deltas = (actions14 + 1.0) * 0.5 * rng + q01

        out = np.zeros((T, 14), dtype=np.float32)
        for arm_in_slice, base_off in (
            (LEFT_ARM_SLICE, 0),
            (RIGHT_ARM_SLICE, 7),
        ):
            # State arrives in so3 (xyz + r6d) layout — convert to xyz+euler.
            arm_state_xyz_eul = _so3_to_xyz_euler_np(state34[arm_in_slice])  # (6,)
            arm_delta_xyz = deltas[:, base_off : base_off + 3]                # (T, 3)
            arm_delta_eul = deltas[:, base_off + 3 : base_off + 6]            # (T, 3)
            # Cumulative roll-out: position adds, orientation adds in Euler
            # space (small per-step angles, train-time convention).
            out[:, base_off : base_off + 3] = arm_state_xyz_eul[:3] + np.cumsum(arm_delta_xyz, axis=0)
            out[:, base_off + 3 : base_off + 6] = arm_state_xyz_eul[3:] + np.cumsum(arm_delta_eul, axis=0)
            # Gripper: rescale [-1, 1] -> [0, 100]
            out[:, base_off + 6] = ((deltas[:, base_off + 6] + 1.0) * 50.0).clip(0.0, 100.0)
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seeded RNGs to %d", seed)


def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        _seed_everything(args.seed)

    policy = QwenPIKMSyncPolicy(
        ckpt_path=args.ckpt_path,
        stats_path=args.stats_path,
        robot_type=args.robot_type,
        default_prompt=args.default_prompt,
        image_height=args.image_height,
        image_width=args.image_width,
        train_freq=args.train_freq,
        use_bf16=args.use_bf16,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating sync server on %s (host=%s ip=%s)", args.port, hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=policy.metadata,
    )
    logger.info("server running ... metadata=%s", policy.metadata)
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to QwenPIKM checkpoint dir")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--idle_timeout", type=int, default=1800)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--robot_type", type=str, default="S1-stationary")
    p.add_argument("--stats_path", type=str, default="./clap/assets/dataset_statistics_32.json")
    p.add_argument("--default_prompt", type=str, default="")
    p.add_argument("--image_height", type=int, default=240)
    p.add_argument("--image_width", type=int, default=320)
    p.add_argument("--train_freq", type=int, default=30)
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(build_argparser().parse_args())
