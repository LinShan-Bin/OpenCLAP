# Copyright 2025 CLAP Team. All rights reserved.
# Licensed under the MIT License.
"""Synchronous websocket inference **client** for a real Astribot S1.

Faithful port of ``latent_action_model/vla_scripts/deploy_astribot.py`` from
the upstream UniVLA codebase, with the following adjustments:

  1. Talk to OpenCLAP's :mod:`sync_policy_server` instead of UniVLA's
     deploy server. The websocket protocol is identical (msgpack-numpy
     handshake + obs/action round-trip), only the obs key names and the
     14-dim CLAP action layout are CLAP-specific.
  2. Insert a ``pdb.set_trace()`` immediately before
     :func:`astribot.move_cartesian_waypoints` so a human operator can
     inspect the predicted waypoints in r6d/quaternion space and confirm
     they're inside the workspace before the robot moves. **Comment out
     the breakpoint after you've verified the policy is safe on your
     setup.** Leaving it in is the recommended default for the very first
     run on a new robot or task.
  3. Replace ``openpi_client.msgpack_numpy`` with the OpenCLAP-bundled
     :mod:`deployment.model_server.tools.msgpack_numpy` (binary-compatible).

ROS / Astribot SDK are imported lazily so that this module can be loaded
(e.g. in unit tests, on workstations) without the robot toolchain.

Usage
-----
::

    python -m examples.Astribot.eval_files.sync_policy_client \\
        --server_host 10.0.0.42 --server_port 8000 \\
        --task "make a bouquet using the red heart and yellow sunflower" \\
        --start_pose_file ./clap/assets/pack_doll.parquet \\
        --duration 0.5 --max_steps 200
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# --- OpenCLAP imports -------------------------------------------------
# Use the bundled msgpack-numpy (binary-compatible with openpi_client's).
from deployment.model_server.tools import msgpack_numpy
import websockets.sync.client


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation helpers — verbatim from deploy_astribot.py.
# ---------------------------------------------------------------------------
def quat_to_r6d(quat: np.ndarray) -> np.ndarray:
    """[qx,qy,qz,qw] (scipy convention) -> [6] r6d (first two rows of R)."""
    rot_mat = R.from_quat(quat).as_matrix()
    return np.concatenate([rot_mat[0, :], rot_mat[1, :]])


def r6d_to_rotation_matrix(r6d: np.ndarray) -> np.ndarray:
    """[..., 6] -> [..., 3, 3] via Gram-Schmidt."""
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2)


def r6d_to_quat(r6d: np.ndarray) -> np.ndarray:
    return R.from_matrix(r6d_to_rotation_matrix(r6d)).as_quat()


# ---------------------------------------------------------------------------
# Camera subscriber (mirrors RGBDRead in deploy_astribot.py)
# ---------------------------------------------------------------------------
class RGBDRead:
    """Subscribe to head / torso / left-wrist / right-wrist RGB topics."""

    def __init__(self, astribot, policy_type: str = "clap") -> None:
        # Lazy imports so `import sync_policy_client` works without ROS.
        import rospy
        from sensor_msgs.msg import CompressedImage
        import cv2

        self._cv2 = cv2
        self.astribot = astribot
        self.policy_type = policy_type
        self.timeout = 100.0
        self.head_rgb: Optional[np.ndarray] = None
        self.torso_image: Optional[np.ndarray] = None
        self.left_image: Optional[np.ndarray] = None
        self.right_image: Optional[np.ndarray] = None
        self.last_head_rgb_time = 0.0
        self.last_torso_time = 0.0
        self.last_left_time = 0.0
        self.last_right_time = 0.0

        rospy.Subscriber(
            "/astribot_camera/head_rgbd/color_compress/compressed",
            CompressedImage, self._head_rgb_cb)
        rospy.Subscriber(
            "/astribot_camera/torso_rgbd/color_compress/compressed",
            CompressedImage, self._torso_cb)
        rospy.Subscriber(
            "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
            CompressedImage, self._left_cb)
        rospy.Subscriber(
            "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
            CompressedImage, self._right_cb)
        print(f"\033[94m[RGBDRead] subscribed for policy={policy_type}\033[0m")

    # -- callbacks ------------------------------------------------------
    def _decode(self, msg) -> np.ndarray:
        np_arr = np.frombuffer(msg.data, np.uint8)
        return self._cv2.imdecode(np_arr, self._cv2.IMREAD_COLOR)

    def _head_rgb_cb(self, msg):
        self.head_rgb = self._decode(msg); self.last_head_rgb_time = time.time()

    def _torso_cb(self, msg):
        self.torso_image = self._decode(msg); self.last_torso_time = time.time()

    def _left_cb(self, msg):
        self.left_image = self._decode(msg); self.last_left_time = time.time()

    def _right_cb(self, msg):
        self.right_image = self._decode(msg); self.last_right_time = time.time()

    # -- public ---------------------------------------------------------
    def get_rgbd(self) -> Optional[Dict[str, np.ndarray]]:
        """Return {head, torso, left, right} or None if any stream timed out."""
        now = time.time()
        for name, img, last in (
            ("head", self.head_rgb, self.last_head_rgb_time),
            ("torso", self.torso_image, self.last_torso_time),
            ("left", self.left_image, self.last_left_time),
            ("right", self.right_image, self.last_right_time),
        ):
            if img is None or now - last > self.timeout:
                print(f"\033[91m[RGBDRead] {name} timeout\033[0m")
                return None
        return {
            "rgb": self.head_rgb.copy(),
            "torso": self.torso_image.copy(),
            "left": self.left_image.copy(),
            "right": self.right_image.copy(),
        }

    def get_rgb_obs_dict(self, astribot_names: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """Get current RGB obs + 20-dim dual-arm pose state in r6d format.

        Robot SDK returns: [left_xyz(3), left_quat(4), left_grip(1),
                            right_xyz(3), right_quat(4), right_grip(1)]   (16 dims)
        We convert to:    [left_xyz(3), left_r6d(6), left_grip(1),
                            right_xyz(3), right_r6d(6), right_grip(1)]    (20 dims)
        """
        poses = np.concatenate(self.astribot.get_current_cartesian_pose(astribot_names))
        left_xyz, left_quat, left_grip = poses[0:3], poses[3:7], poses[7:8]
        right_xyz, right_quat, right_grip = poses[8:11], poses[11:15], poses[15:16]
        pose_state = np.concatenate([
            left_xyz, quat_to_r6d(left_quat), left_grip,
            right_xyz, quat_to_r6d(right_quat), right_grip,
        ])  # (20,)
        rgbd = self.get_rgbd()
        if rgbd is None:
            return None
        out = {"pose_state": pose_state}
        out.update(rgbd)
        return out


def go_to_init_pose(astribot, local_data_path: str) -> None:
    """Move robot to the initial pose stored in an hdf5 / npy / parquet file."""
    ext = os.path.splitext(local_data_path)[1].lower()
    if ext == ".hdf5":
        import h5py
        with h5py.File(local_data_path, "r") as root:
            jp = (root["/joints_dict/joints_position_command"][()][0]
                  if "joints_dict" in root else
                  root["/joints_dict_time_align/joints_position_command"][()][0])
    elif ext == ".npy":
        jp = np.load(local_data_path)
    elif ext == ".parquet":
        import pandas as pd
        jp = pd.read_parquet(local_data_path)["joints_dict.joints_position_command"][0]
    else:
        raise ValueError(f"Unsupported start_pose_file extension: {ext}")
    jp_split = np.split(jp, np.cumsum(astribot.whole_body_dofs[:]))[:-1]
    jp_lists = [j.tolist() for j in jp_split]
    astribot.move_joints_position(
        astribot.whole_body_names[1:], jp_lists[1:], duration=3
    )


# ---------------------------------------------------------------------------
# CLAP websocket policy client (compatible with WebsocketPolicyServer)
# ---------------------------------------------------------------------------
class ClapPolicyClient:
    """Websocket client for the OpenCLAP sync policy server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.host = host
        self.port = port
        self.packer = msgpack_numpy.Packer()
        self.ws = None
        self.metadata: Optional[Dict] = None
        self._connect()

    def _connect(self) -> None:
        uri = self.host if self.host.startswith(("ws://", "wss://")) else f"ws://{self.host}"
        if self.port is not None:
            uri += f":{self.port}"
        print(f"[ClapPolicyClient] connecting to {uri}")
        self.ws = websockets.sync.client.connect(
            uri, compression=None, max_size=None, ping_interval=None, ping_timeout=60,
        )
        self.metadata = msgpack_numpy.unpackb(self.ws.recv())
        print(f"[ClapPolicyClient] connected. metadata={self.metadata}")

    def close(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def reset(self, task: str) -> None:
        # The server is stateless across requests; reset is a no-op.
        print(f"[ClapPolicyClient] reset task={task!r}")

    def step(
        self,
        head_image: np.ndarray,
        task: str,
        state_r6d: np.ndarray,
        wrist_left: Optional[np.ndarray] = None,
        wrist_right: Optional[np.ndarray] = None,
    ) -> Dict:
        """Send one observation, receive an action chunk.

        Args:
            head_image: HxWxC uint8 RGB array
            task: instruction string
            state_r6d: 20-dim dual-arm state in (xyz, r6d, grip) layout
            wrist_left/right: optional HxWxC uint8 RGB arrays
        """
        if head_image.ndim == 3 and head_image.shape[-1] == 3:
            head_chw = np.transpose(head_image, (2, 0, 1))
        else:
            head_chw = head_image

        # Pad 20-dim arm state into the 34-dim cartesian_pose layout used at
        # train time (see clap/data_transform.py:KeyMapping). Indices 9..29
        # are the dual arms; everything else is filled with zeros and ignored
        # by the server's _build_example.
        padded_state = np.zeros(34, dtype=np.float32)
        padded_state[9:29] = state_r6d.astype(np.float32)

        obs = {
            "images_dict.head.rgb": head_chw,
            "task": task,
            "cartesian_so3_dict.cartesian_pose_state": padded_state,
        }
        if wrist_left is not None:
            obs["images_dict.left.rgb"] = np.transpose(wrist_left, (2, 0, 1))
        if wrist_right is not None:
            obs["images_dict.right.rgb"] = np.transpose(wrist_right, (2, 0, 1))

        t0 = time.time()
        self.ws.send(self.packer.pack(obs))
        resp = self.ws.recv()
        if isinstance(resp, str):
            return {"status": "error", "message": resp,
                    "action": None, "inference_time": time.time() - t0}
        out = msgpack_numpy.unpackb(resp)
        action = out.get("actions", out.get("action"))
        if action is None:
            return {"status": "error", "message": "no actions in response",
                    "action": None, "inference_time": time.time() - t0}
        return {
            "status": "success",
            "action": np.asarray(action),
            "inference_time": time.time() - t0,
            "obs_timestamp": out.get("obs_timestamp"),
        }


# ---------------------------------------------------------------------------
# Main control loop (mirrors deploy_astribot.main)
# ---------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server_host", default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--task", required=True, help="Instruction for the robot")
    p.add_argument("--duration", type=float, default=0.5,
                   help="Per-step duration (seconds) within the action chunk.")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Maximum number of inference loops.")
    p.add_argument("--head_follow_flag", type=lambda s: s.lower() == "true",
                   default=False, help="Robot head follows the end-effector.")
    p.add_argument("--start_pose_file", default="./clap/assets/pack_doll.parquet",
                   help="Initial joint pose file (.parquet/.hdf5/.npy).")
    p.add_argument("--use_wrist_camera", action="store_true",
                   help="Send left+right wrist RGB to the server.")
    return p.parse_args()


def main() -> None:
    args = get_args()
    # Lazy ROS / SDK imports so the module is importable without them.
    import rospy
    import cv2
    from core.astribot_api.astribot_client import Astribot

    astribot = Astribot(high_control_rights=True)
    rgbd_iface = RGBDRead(astribot, policy_type="clap")

    print("[client] waiting for sensor frames…")
    while rgbd_iface.get_rgbd() is None:
        time.sleep(0.5)
        print("[client] still waiting…")
    print("[client] sensor data online.")

    print(f"[client] connecting to policy server {args.server_host}:{args.server_port}")
    client = ClapPolicyClient(host=args.server_host, port=args.server_port)
    client.reset(args.task)

    print("[client] moving to init pose…")
    astribot.set_head_follow_effector_old(args.head_follow_flag)
    go_to_init_pose(astribot, args.start_pose_file)
    print("[client] init pose reached.")
    input("[client] press <Enter> to start the control loop…")

    arm_joint_names = [
        "astribot_arm_left", "astribot_gripper_left",
        "astribot_arm_right", "astribot_gripper_right",
    ]
    obs_queue: deque = deque(maxlen=1)
    while not obs_queue:
        obs0 = rgbd_iface.get_rgb_obs_dict(arm_joint_names)
        if obs0 is not None:
            obs_queue.append(obs0)
        else:
            print("[client] obs not ready; retrying…")
            time.sleep(1)

    rate = rospy.Rate(1.0 / args.duration)
    loop = 0
    print("[client] entering control loop…")
    while not rospy.is_shutdown() and loop < args.max_steps:
        obs = obs_queue[0]
        head_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB)
        cur_pose_r6d = obs["pose_state"]
        wrist_left = wrist_right = None
        if args.use_wrist_camera:
            wrist_left = cv2.cvtColor(obs["left"], cv2.COLOR_BGR2RGB)
            wrist_right = cv2.cvtColor(obs["right"], cv2.COLOR_BGR2RGB)

        t_call = time.time()
        resp = client.step(head_rgb, args.task, cur_pose_r6d, wrist_left, wrist_right)
        if resp["status"] != "success":
            print(f"\033[91m[client] server error: {resp.get('message')}\033[0m")
            break

        action_chunk = np.asarray(resp["action"])              # (N, ?)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        if action_chunk.ndim == 3:                             # (1, N, ?) -> (N, ?)
            action_chunk = action_chunk[0]
        N = action_chunk.shape[0]
        print(f"[client] loop={loop}  infer={resp['inference_time']*1000:.1f}ms  "
              f"chunk={action_chunk.shape}")

        # ---- Convert action chunk to dual-arm waypoints --------------
        # The server returns absolute per-arm poses in xyz_rpy (Euler) layout:
        #   per arm: [xyz(3), euler(3), gripper(1)]  → 7 dims.  Total: 14.
        # The Astribot SDK takes (xyz, quat, gripper) per arm. We therefore
        # convert each Euler triplet to a quaternion via scipy.
        APD = 7   # action_dim_per_arm in xyz_rpy
        assert action_chunk.shape[1] >= 2 * APD, \
            f"expected per-step action width >= {2*APD}, got {action_chunk.shape[1]}"

        waypoints: List[List[np.ndarray]] = []
        for k in range(N):
            la = action_chunk[k, :APD]
            ra = action_chunk[k, APD:2 * APD]
            l_xyz, l_eul, l_grip = la[:3], la[3:6], la[6:7]
            r_xyz, r_eul, r_grip = ra[:3], ra[3:6], ra[6:7]
            l_quat = R.from_euler("xyz", l_eul).as_quat()
            r_quat = R.from_euler("xyz", r_eul).as_quat()
            l_arm = np.concatenate([l_xyz, l_quat]).astype(np.float64)
            r_arm = np.concatenate([r_xyz, r_quat]).astype(np.float64)
            waypoints.append([
                l_arm, l_grip.astype(np.float64),
                r_arm, r_grip.astype(np.float64),
            ])

        time_list = [(i + 1) * args.duration for i in range(N)]

        # =====================================================================
        # SAFETY CHECK — KEEP THIS pdb BREAKPOINT UNTIL YOU'VE VERIFIED THE
        # POLICY IS SAFE ON YOUR ROBOT. Inspect ``waypoints`` at the prompt:
        #
        #   (Pdb) print(waypoints[0])     # first per-step [l_arm, l_grip, r_arm, r_grip]
        #   (Pdb) print(np.array([w[0] for w in waypoints]))   # all left-arm xyz/quat
        #   (Pdb) print(np.array([w[2] for w in waypoints]))   # all right-arm xyz/quat
        #
        # Confirm:
        #   * xyz stays inside the workspace box you trust
        #   * the chunk's first xyz is close to ``cur_pose_r6d[0:3]`` /
        #     ``cur_pose_r6d[10:13]`` (otherwise the very first move will be a
        #     large jump)
        #   * grippers are in [0, 100] and not flipping wildly between steps
        #
        # When you're confident, COMMENT OUT the next line. Do **not** delete
        # it — leave it in place so the next operator on a new task knows to
        # re-enable the safety stop.
        breakpoint()
        # =====================================================================

        astribot.move_cartesian_waypoints(
            arm_joint_names, waypoints, time_list, use_wbc=True,
        )

        # Refresh observation for the next inference loop.
        new_obs = rgbd_iface.get_rgb_obs_dict(arm_joint_names)
        if new_obs is not None:
            obs_queue.append(new_obs)
        else:
            print("\033[91m[client] failed to refresh obs; reusing previous\033[0m")
        loop += 1

    print("[client] loop ended; closing client.")
    client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
