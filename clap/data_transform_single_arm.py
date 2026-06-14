import json
from typing import Dict, Any, List

import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from clap.unified_action import ensure_arm_mask, make_arm_mask, zero_inactive_slots
from pathlib import Path as _Path
_CLAP_ROOT = _Path(__file__).resolve().parent
def _resolve(p):
    p = str(p)
    return p if _Path(p).is_absolute() else str(_CLAP_ROOT / p.lstrip('./'))




KEY_MAPPING = {
    # Agibot format
    "cartesian_so3_dict.cartesian_pose_command": "action",
    "cartesian_so3_dict.cartesian_pose_state": "state",
    "images_dict.head.rgb": "observation.head",
    "images_dict.right.rgb": "observation.right",
    "images_dict.left.rgb": "observation.left",
    # Bridge format
    "action": "action",
    "observation.state": "state",
    "observation.images.image_0": "observation.head",
    "observation.images.image_1": "observation.left",
    "observation.images.image_2": "observation.right",
    # Libero format
    "observation.images.image": "observation.head",
    # DROID format
    "observation.images.droid_external": "observation.head",
    "observation.images.wrist_image_left": "observation.right",
    # Common fields
    "task": "task",
    "subtask": "subtask",
    "dataset_index": "dataset_index",
    "dataset_name": "dataset_name",
    "robot_type": "robot_type",
    "robot_id": "robot_id",
}


class KeyMapping(nn.Module):
    def __init__(self):
        super(KeyMapping, self).__init__()
        self.mapping = KEY_MAPPING

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        ret_dict = {}
        for k, v in trajectory.items():
            if k in self.mapping:
                new_key = self.mapping[k]
            else:
                new_key = k
            ret_dict[new_key] = v
        return ret_dict


class ActionStateIndex(nn.Module):
    def __init__(self, action_index: List[int] = [0, 1, 2, 3, 4, 5, 6], state_index: List[int] = [0, 1, 2, 3, 4, 5, 6]):
        super(ActionStateIndex, self).__init__()
        self.action_index = action_index
        self.state_index = state_index
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory["action"] = trajectory["action"][..., self.action_index]
        trajectory["state"] = trajectory["state"][..., self.state_index]
        return trajectory


class ToRPY(nn.Module):
    def __init__(self, apply_to: List[str] = ["state"]):
        super(ToRPY, self).__init__()
        self.apply_to = apply_to
    
    @staticmethod
    def quat_to_rpy(quat):
        original_shape = quat.shape
        # Flatten to (N, 4) for scipy
        quat_np = quat.detach().cpu().numpy().reshape(-1, 4)
        rpy_np = R.from_quat(quat_np).as_euler('xyz')
        # Reshape back to original dimensions with 3 instead of 4
        rpy_np = rpy_np.reshape(original_shape[:-1] + (3,))
        return torch.from_numpy(rpy_np).to(quat)
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.apply_to:
            # Convert quaternion (4D) to RPY (3D)
            # Input:  [..., x, y, z, qx, qy, qz, qw, gripper] (8D)
            # Output: [..., x, y, z, roll, pitch, yaw, gripper] (7D)
            quat = trajectory[key][..., 3:7]  # Extract quaternion (4D)
            rpy = self.quat_to_rpy(quat)       # Convert to RPY (3D)
            
            # Rebuild the tensor with RPY instead of quaternion
            trajectory[key] = torch.cat([
                trajectory[key][..., :3],   # x, y, z
                rpy,                         # roll, pitch, yaw
                trajectory[key][..., 7:]    # gripper (and any following dims)
            ], dim=-1)
        
        return trajectory


class GripperMapping(nn.Module):
    """Map gripper values from [0,1] (closed to open) to [1,-1] (closed to open).
    
    Input [0,1]: 0=closed, 1=open
    Output [1,-1]: 1=closed, -1=open
    Formula: output = 1 - 2 * input
    """
    def __init__(self):
        super(GripperMapping, self).__init__()
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory["action"][..., 6] = 1 - 2 * trajectory["action"][..., 6]
        trajectory["state"][..., 6] = 1 - 2 * trajectory["state"][..., 6]
        return trajectory


class DroidStyleGripperMapping(nn.Module):
    """Map gripper values from [0,1] (open to closed) to [-1,1] (open to closed)."""

    def __init__(self):
        super(DroidStyleGripperMapping, self).__init__()

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory["action"][..., 6] = (2 * trajectory["action"][..., 6] - 1).clamp(-1, 1)
        trajectory["state"][..., 6] = (2 * trajectory["state"][..., 6] - 1).clamp(-1, 1)
        return trajectory


class ActionCumsum(nn.Module):
    def __init__(self):
        super(ActionCumsum, self).__init__()
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        action = torch.cumsum(trajectory["action"], dim=-2)
        trajectory["action"][..., :6] = action[..., :6]  # Without gripper
        return trajectory


class Padding(nn.Module):
    def __init__(self, arm_layout: str = "single_right"):
        super(Padding, self).__init__()
        # Make the action compatible with dual arm actions
        self.arm_layout = arm_layout
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        action = trajectory["action"]
        batch_size = action.shape[0]
        arm_mask = make_arm_mask(
            trajectory.get("arm_layout", self.arm_layout),
            batch_size,
            default_layout=self.arm_layout,
            device=action.device,
            dtype=torch.bool,
        )
        action_slots = torch.zeros(
            *action.shape[:-1],
            2,
            action.shape[-1],
            dtype=action.dtype,
            device=action.device,
        )
        if arm_mask[:, 0].any():
            action_slots[arm_mask[:, 0], :, 0, :] = action[arm_mask[:, 0]]
        if arm_mask[:, 1].any():
            action_slots[arm_mask[:, 1], :, 1, :] = action[arm_mask[:, 1]]
        trajectory["action"] = action_slots.flatten(-2)
        trajectory["action_static_reference"] = trajectory["action"].clone()
        
        if "state" in trajectory:
            state = trajectory["state"]
            state_slots = torch.zeros(
                *state.shape[:-1],
                2,
                state.shape[-1],
                dtype=state.dtype,
                device=state.device,
            )
            if arm_mask[:, 0].any():
                state_slots[arm_mask[:, 0], :, 0, :] = state[arm_mask[:, 0]]
            if arm_mask[:, 1].any():
                state_slots[arm_mask[:, 1], :, 1, :] = state[arm_mask[:, 1]]
            trajectory["state"] = state_slots.flatten(-2)

        trajectory["arm_mask"] = arm_mask.to(dtype=torch.float32)
        trajectory["arm_layout"] = [
            "single_left" if bool(mask[0]) else "single_right"
            for mask in arm_mask.cpu()
        ]
        
        return trajectory


class Unpadding(nn.Module):
    def __init__(self):
        super(Unpadding, self).__init__()
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory["action"] = trajectory["action"][..., :7]
        if "state" in trajectory:
            trajectory["state"] = trajectory["state"][..., :7]
        return trajectory


class LiberoStateToAction(nn.Module):
    """Build LIBERO action chunks from state windows using the DROID state-delta convention.

    LIBERO state format used here is:
    [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper, ...].

    Each action is relative to the first state in the sampled chunk:
    action[t, :3] = state[t + 1, :3] - state[0, :3]
    action[t, 3:6] = euler_xyz(R(state[t + 1]) @ R(state[0]).T)
    action[t, 6] = state[t + 1, gripper]
    """

    def __init__(self, gripper_index: int = 6):
        super(LiberoStateToAction, self).__init__()
        self.gripper_index = gripper_index

    @staticmethod
    def rotvec_relative_to_rpy(base_rotvec: torch.Tensor, target_rotvec: torch.Tensor) -> torch.Tensor:
        base_shape = base_rotvec.shape
        base_np = base_rotvec.detach().cpu().numpy().reshape(-1, 3)
        target_np = target_rotvec.detach().cpu().numpy().reshape(-1, 3)

        base_mat = R.from_rotvec(base_np).as_matrix()
        target_mat = R.from_rotvec(target_np).as_matrix()
        delta_mat = target_mat @ base_mat.transpose(0, 2, 1)
        rpy_np = R.from_matrix(delta_mat).as_euler("xyz")
        return torch.from_numpy(rpy_np.reshape(base_shape)).to(base_rotvec)

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        state = trajectory["state"]
        if state.shape[-1] <= self.gripper_index:
            raise ValueError(f"LIBERO state must contain gripper index {self.gripper_index}, got {state.shape[-1]}")
        if state.shape[-2] < 2:
            raise ValueError("LIBERO state sequence must contain at least two frames")

        state = torch.cat(
            [
                state[..., :6],
                state[..., self.gripper_index:self.gripper_index + 1],
            ],
            dim=-1,
        )
        base = state[..., :1, :]
        target = state[..., 1:, :]

        action = target.clone()
        action[..., :3] = target[..., :3] - base[..., :3]
        base_rotvec = base[..., 3:6].expand_as(target[..., 3:6])
        action[..., 3:6] = self.rotvec_relative_to_rpy(base_rotvec, target[..., 3:6])

        trajectory["action"] = action
        trajectory["state"] = base
        return trajectory


class RobotIdOverride(nn.Module):
    def __init__(self, robot_id: int | None = None):
        super(RobotIdOverride, self).__init__()
        self.robot_id = robot_id

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        if self.robot_id is None:
            return trajectory
        reference = trajectory.get("action", trajectory.get("state"))
        if not isinstance(reference, torch.Tensor):
            raise ValueError("RobotIdOverride requires an action or state tensor")
        trajectory["robot_id"] = torch.full(
            (reference.shape[0],),
            int(self.robot_id),
            dtype=torch.long,
            device=reference.device,
        )
        return trajectory


class ActionDiff(nn.Module):
    def __init__(self):
        super(ActionDiff, self).__init__()
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        action = trajectory["action"]
        action[..., 1:, :] = action[..., 1:, :] - action[..., :-1, :]
        trajectory["action"][..., :6] = action[..., :6]  # Without gripper
        return trajectory


class ActionNormalization(nn.Module):
    def __init__(self, stats_path: str = _resolve("assets/dataset_statistics_32.json")):
        super(ActionNormalization, self).__init__()
        dataset_stats = json.load(open(stats_path))
        astribot_stats = dataset_stats["S1-stationary"]
        agibot_stats = dataset_stats["agibot-go1"]
        libero_stats = dataset_stats["franka"]
        fractal_stats = dataset_stats["fractal"]
        bridge_stats = dataset_stats["widowx"]
        droid_stats = dataset_stats.get("droid", libero_stats)
        
        action_q99 = torch.stack([
            torch.tensor(astribot_stats["action"]["q99"], dtype=torch.float32),
            torch.tensor(agibot_stats["action"]["q99"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
            torch.tensor(libero_stats["action"]["q99"], dtype=torch.float32),
            torch.tensor(fractal_stats["action"]["q99"], dtype=torch.float32),
            torch.tensor(bridge_stats["action"]["q99"], dtype=torch.float32),
            torch.tensor(droid_stats["action"]["q99"], dtype=torch.float32),
        ])
        action_q01 = torch.stack([
            torch.tensor(astribot_stats["action"]["q01"], dtype=torch.float32),
            torch.tensor(agibot_stats["action"]["q01"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
            torch.tensor(libero_stats["action"]["q01"], dtype=torch.float32),
            torch.tensor(fractal_stats["action"]["q01"], dtype=torch.float32),
            torch.tensor(bridge_stats["action"]["q01"], dtype=torch.float32),
            torch.tensor(droid_stats["action"]["q01"], dtype=torch.float32),
        ])
        self.register_buffer("action_q99", action_q99)
        self.register_buffer("action_q01", action_q01)
        
        state_q99 = torch.stack([
            torch.tensor(astribot_stats["state"]["q99"], dtype=torch.float32),
            torch.tensor(agibot_stats["state"]["q99"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
            torch.tensor(libero_stats["state"]["q99"], dtype=torch.float32),
            torch.tensor(fractal_stats["state"]["q99"], dtype=torch.float32),
            torch.tensor(bridge_stats["state"]["q99"], dtype=torch.float32),
            torch.tensor(droid_stats["state"]["q99"], dtype=torch.float32),
        ])
        state_q01 = torch.stack([
            torch.tensor(astribot_stats["state"]["q01"], dtype=torch.float32),
            torch.tensor(agibot_stats["state"]["q01"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
            torch.tensor(libero_stats["state"]["q01"], dtype=torch.float32),
            torch.tensor(fractal_stats["state"]["q01"], dtype=torch.float32),
            torch.tensor(bridge_stats["state"]["q01"], dtype=torch.float32),
            torch.tensor(droid_stats["state"]["q01"], dtype=torch.float32),
        ])
        self.register_buffer("state_q99", state_q99)
        self.register_buffer("state_q01", state_q01)
        
        normalize_mask = torch.tensor(
            [True] * 6 + [False] + [True] * 6 + [False], dtype=torch.bool)
        self.register_buffer("normalize_mask", normalize_mask)
        
        print("Loaded action normalization statistics")

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        action = trajectory["action"]
        action_q99 = self.action_q99[trajectory["robot_id"]][:, None]
        action_q01 = self.action_q01[trajectory["robot_id"]][:, None]
        action_normalized = (action - action_q01) / (action_q99 - action_q01 + 1e-8) * 2 - 1
        action[:, :, self.normalize_mask] = action_normalized[:, :, self.normalize_mask]
        trajectory["action"] = action
        
        state = trajectory["state"]
        state_q99 = self.state_q99[trajectory["robot_id"]][:, None]
        state_q01 = self.state_q01[trajectory["robot_id"]][:, None]
        state_normalized = (state - state_q01) / (state_q99 - state_q01 + 1e-8) * 2 - 1
        state[:, :, self.normalize_mask] = state_normalized[:, :, self.normalize_mask]
        trajectory["state"] = state
        
        return trajectory


class ApplyArmMask(nn.Module):
    def __init__(self, action_dim_per_arm: int = 7, default_layout: str = "single_right"):
        super(ApplyArmMask, self).__init__()
        self.action_dim_per_arm = action_dim_per_arm
        self.default_layout = default_layout

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory = ensure_arm_mask(trajectory, default_layout=self.default_layout)
        arm_mask = trajectory["arm_mask"].to(dtype=torch.bool)
        trajectory["action"] = zero_inactive_slots(trajectory["action"], arm_mask, self.action_dim_per_arm)
        if "state" in trajectory:
            trajectory["state"] = zero_inactive_slots(trajectory["state"], arm_mask, self.action_dim_per_arm)
        return trajectory


class SingleArmPipeline(nn.Module):
    def __init__(self,
                 action_index: List[int] = [0, 1, 2, 3, 4, 5, 6],
                 state_index: List[int] = [0, 1, 2, 3, 4, 5, 6],
                 apply_to: List[str] = ["state"],
                 arm_layout: str = "single_right",
                 normalization_stats_path: str = _resolve("assets/dataset_statistics_32.json")):
        super(SingleArmPipeline, self).__init__()
        
        self.pipeline = nn.Sequential(
            KeyMapping(),
            ActionStateIndex(action_index=action_index, state_index=state_index),
            ToRPY(apply_to=apply_to),
            GripperMapping(),
            # ActionCumsum(),
            Padding(arm_layout=arm_layout),
            ActionNormalization(stats_path=normalization_stats_path),
            ApplyArmMask(action_dim_per_arm=7, default_layout=arm_layout),
        )

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory = self.pipeline(trajectory)
        return trajectory


class LiberoDroidStyleTokenPipeline(nn.Module):
    def __init__(
        self,
        stats_path: str = _resolve("assets/dataset_statistics_32.json"),
        normalization_robot_id: int | None = None,
        arm_layout: str = "single_right",
        gripper_index: int = 6,
    ):
        super(LiberoDroidStyleTokenPipeline, self).__init__()

        self.pipeline = nn.Sequential(
            KeyMapping(),
            LiberoStateToAction(gripper_index=gripper_index),
            DroidStyleGripperMapping(),
            Padding(arm_layout=arm_layout),
            RobotIdOverride(normalization_robot_id),
            ActionNormalization(stats_path=stats_path),
            ApplyArmMask(action_dim_per_arm=7, default_layout=arm_layout),
        )

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipeline(trajectory)
