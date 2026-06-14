# Copyright 2025 starVLA community.
"""Shared Astribot tensor transforms.

The single place where the
``KeyMapping → ActionDelta → ActionNormalization`` pipeline lives, imported by:

  * ``examples/Astribot/train_files/data_registry/data_config.py``
    (the per-sample post-pack hook on top of starVLA's ``LeRobotSingleDataset``)
  * ``examples/Astribot/eval_files/async_policy_server.py``
    (state preprocessing + delta-to-absolute reconstruction)

Keep it self-contained — no torch / no PyArrow imports — so it can be
exercised inside a DataLoader worker without surprise side effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Robot-id mapping (kept in sync with clap/custom_lerobot.py).
# ---------------------------------------------------------------------------
ROBOT_ID_MAPPING = {
    "S1-stationary": 0,
    "agibot-go1": 1,
    "human": 2,
    "franka": 3,
    "google_robot": 4,
    "widowx": 5,
    "droid": 6,
}

# 34-dim cartesian_pose layout (see model_server_protocol.md §5):
#   [0:9)   torso so3
#   [9:18)  left arm so3   (xyz + r6d, 9 dims)
#   [18]    left gripper (0..100)
#   [19:28) right arm so3
#   [28]    right gripper
#   [29:31) head joints
#   [31:34) chassis joints
LEFT_ARM_SLICE = slice(9, 18)
LEFT_GRIPPER = 18
RIGHT_ARM_SLICE = slice(19, 28)
RIGHT_GRIPPER = 28


# Mask over the 14-dim post-pipeline action: gripper dims are kept in
# [-1, 1] without quantile rescale.
_NORMALIZE_MASK_14 = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=np.bool_)


def _r6d_to_mat_np(r6d: np.ndarray) -> np.ndarray:
    """[..., 6] r6d → [..., 3, 3] rotation matrix (Gram-Schmidt)."""
    r1 = r6d[..., 0:3]
    r2 = r6d[..., 3:6]
    b1 = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-8)
    proj = (b1 * r2).sum(axis=-1, keepdims=True) * b1
    u2 = r2 - proj
    b2 = u2 / (np.linalg.norm(u2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2)


def _mat_to_euler_np(R: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotation matrix → [..., 3] (roll, pitch, yaw) XYZ-Euler."""
    r00 = R[..., 0, 0]
    r10 = R[..., 1, 0]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    pitch_y = np.arcsin(-np.clip(r20, -1.0 + 1e-7, 1.0 - 1e-7))
    roll_x = np.arctan2(r21, r22)
    yaw_z = np.arctan2(r10, r00)
    return np.stack([roll_x, pitch_y, yaw_z], axis=-1)


def _so3_to_xyz_euler_np(so3: np.ndarray) -> np.ndarray:
    """[..., 9] (xyz + r6d) → [..., 6] (xyz + euler)."""
    xyz = so3[..., :3]
    R = _r6d_to_mat_np(so3[..., 3:])
    euler = _mat_to_euler_np(R)
    return np.concatenate([xyz, euler], axis=-1)


def _delta_xyz_euler_np(action_arm: np.ndarray, state_arm: np.ndarray) -> np.ndarray:
    """Compute the 6-D xyz+euler delta of two so3-9 vectors.

    state_arm is broadcast over the time dimension (use a 1-step state).
    """
    delta_pos = action_arm[..., :3] - state_arm[..., :3]
    R_act = _r6d_to_mat_np(action_arm[..., 3:])
    R_state = _r6d_to_mat_np(state_arm[..., 3:])
    R_delta = np.matmul(R_act, np.swapaxes(R_state, -1, -2))
    delta_euler = _mat_to_euler_np(R_delta)
    return np.concatenate([delta_pos, delta_euler], axis=-1)


@dataclass
class AstribotStats:
    """Per-robot quantile bounds used for action / state normalization."""

    action_q01: np.ndarray  # (14,)
    action_q99: np.ndarray
    state_q01: np.ndarray
    state_q99: np.ndarray


def load_stats_for_robot(stats_path: Path | str, robot_type: str) -> AstribotStats:
    stats_path = Path(stats_path)
    with open(stats_path, "r") as f:
        all_stats = json.load(f)
    if robot_type not in all_stats:
        raise KeyError(
            f"AstribotStats: robot_type={robot_type!r} not in {stats_path}; "
            f"available keys: {list(all_stats)}"
        )
    s = all_stats[robot_type]
    return AstribotStats(
        action_q01=np.asarray(s["action"]["q01"], dtype=np.float32),
        action_q99=np.asarray(s["action"]["q99"], dtype=np.float32),
        state_q01=np.asarray(s["state"]["q01"], dtype=np.float32),
        state_q99=np.asarray(s["state"]["q99"], dtype=np.float32),
    )


def astribot_pipeline_np(
    raw_action_34: np.ndarray,         # (T, 34) cartesian_pose_command
    raw_state_34: np.ndarray,          # (T_state, 34) cartesian_pose_state
    stats: AstribotStats,
) -> Dict[str, np.ndarray]:
    """KeyMapping ∘ ActionDelta ∘ ActionNormalization on numpy arrays.

    Returns processed ``action`` (T, 14) and ``state`` (T_state, 14). The
    state is delta-friendly per-arm xyz+euler+gripper, the same shape used
    by QwenPIKM's flow-matching head.
    """
    state_ref = raw_state_34[:1]  # (1, 34)
    left_act = raw_action_34[..., LEFT_ARM_SLICE]
    right_act = raw_action_34[..., RIGHT_ARM_SLICE]
    left_grip = raw_action_34[..., LEFT_GRIPPER:LEFT_GRIPPER + 1]
    right_grip = raw_action_34[..., RIGHT_GRIPPER:RIGHT_GRIPPER + 1]

    left_state = raw_state_34[..., LEFT_ARM_SLICE]
    right_state = raw_state_34[..., RIGHT_ARM_SLICE]
    left_grip_state = raw_state_34[..., LEFT_GRIPPER:LEFT_GRIPPER + 1]
    right_grip_state = raw_state_34[..., RIGHT_GRIPPER:RIGHT_GRIPPER + 1]

    delta_left = _delta_xyz_euler_np(left_act, state_ref[..., LEFT_ARM_SLICE])
    delta_right = _delta_xyz_euler_np(right_act, state_ref[..., RIGHT_ARM_SLICE])

    left_state_xyzeuler = _so3_to_xyz_euler_np(left_state)
    right_state_xyzeuler = _so3_to_xyz_euler_np(right_state)

    def _grip_norm(g: np.ndarray) -> np.ndarray:
        # 0..100 → -1..1
        return np.clip(g / 100.0 * 2.0 - 1.0, -1.0, 1.0)

    left_grip = _grip_norm(left_grip)
    right_grip = _grip_norm(right_grip)
    left_grip_state = _grip_norm(left_grip_state)
    right_grip_state = _grip_norm(right_grip_state)

    action_14 = np.concatenate([delta_left, left_grip, delta_right, right_grip], axis=-1)
    state_14 = np.concatenate(
        [left_state_xyzeuler, left_grip_state, right_state_xyzeuler, right_grip_state],
        axis=-1,
    )

    def _normalize(arr: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
        scale = q99 - q01
        scale = np.where(np.abs(scale) < 1e-8, np.ones_like(scale), scale)
        out = arr.copy()
        normalized = (arr - q01) / scale * 2.0 - 1.0
        out[..., _NORMALIZE_MASK_14] = normalized[..., _NORMALIZE_MASK_14]
        return out

    action_14 = _normalize(action_14, stats.action_q01, stats.action_q99)
    state_14 = _normalize(state_14, stats.state_q01, stats.state_q99)

    return {"action": action_14.astype(np.float32), "state": state_14.astype(np.float32)}
