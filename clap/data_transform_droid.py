from typing import Any, Dict

import torch
from torch import nn

from clap.data_transform_single_arm import (
    ActionNormalization,
    ApplyArmMask,
    KeyMapping,
    Padding,
)

from pathlib import Path as _Path
_CLAP_ROOT = _Path(__file__).resolve().parent


def _resolve(p):
    p = str(p)
    return p if _Path(p).is_absolute() else str(_CLAP_ROOT / p.lstrip('./'))


DROID_EXTERNAL_IMAGE_KEYS = (
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
)
DROID_WRIST_IMAGE_KEY = "observation.images.wrist_image_left"
DROID_SELECTED_EXTERNAL_KEY = "observation.images.droid_external"


class DroidStateToAction(nn.Module):
    """Build DROID action chunks from state windows.

    DROID's stored action field is already post-processed and should not be used
    for CLAP/VLA supervision. The state format is:
    [x, y, z, roll, pitch, yaw, pad, gripper].

    Each action in a sampled chunk is relative to the first state in that chunk,
    matching the dual-arm ActionDelta convention.
    """

    @staticmethod
    def rpy_to_mat(rpy: torch.Tensor) -> torch.Tensor:
        """Convert XYZ Euler angles to rotation matrices using R = Rz @ Ry @ Rx."""
        roll = rpy[..., 0]
        pitch = rpy[..., 1]
        yaw = rpy[..., 2]

        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        row0 = torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1)
        row1 = torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1)
        row2 = torch.stack([-sp, cp * sr, cp * cr], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def mat_to_euler(rotation_matrix: torch.Tensor) -> torch.Tensor:
        r00 = rotation_matrix[..., 0, 0]
        r10 = rotation_matrix[..., 1, 0]
        r20 = rotation_matrix[..., 2, 0]
        r21 = rotation_matrix[..., 2, 1]
        r22 = rotation_matrix[..., 2, 2]

        pitch_y = torch.asin(-r20.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        roll_x = torch.atan2(r21, r22)
        yaw_z = torch.atan2(r10, r00)
        return torch.stack([roll_x, pitch_y, yaw_z], dim=-1)

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        state = trajectory["state"]
        if state.shape[-1] < 8:
            raise ValueError(f"DROID state must have at least 8 dims, got {state.shape[-1]}")
        if state.shape[-2] < 2:
            raise ValueError("DROID state sequence must contain at least two frames")

        state = state[..., [0, 1, 2, 3, 4, 5, 7]]
        action = state[..., 1:, :].clone()
        action[..., :3] = state[..., 1:, :3] - state[..., :1, :3]

        rotation_next = self.rpy_to_mat(state[..., 1:, 3:6])
        rotation_current = self.rpy_to_mat(state[..., :1, 3:6])
        delta_rotation = torch.matmul(rotation_next, rotation_current.transpose(-2, -1))
        action[..., 3:6] = self.mat_to_euler(delta_rotation)

        trajectory["action"] = action
        trajectory["state"] = state[..., :1, :]
        return trajectory


class DroidGripperMapping(nn.Module):
    """Map DROID gripper position to the dual-arm canonical convention.

    DROID stores gripper position as 0=open, 1=closed. The dual-arm pipeline
    uses -1=open, +1=closed after mapping raw 0..100 values with x / 100 * 2 - 1.
    """

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory["action"][..., 6] = (2 * trajectory["action"][..., 6] - 1).clamp(-1, 1)
        trajectory["state"][..., 6] = (2 * trajectory["state"][..., 6] - 1).clamp(-1, 1)
        return trajectory


class DroidPipeline(nn.Module):
    def __init__(self, arm_layout: str = "single_right"):
        super(DroidPipeline, self).__init__()
        self.pipeline = nn.Sequential(
            KeyMapping(),
            DroidStateToAction(),
            DroidGripperMapping(),
            Padding(arm_layout=arm_layout),
            ActionNormalization(stats_path=_resolve("assets/dataset_statistics_32.json")),
            ApplyArmMask(action_dim_per_arm=7, default_layout=arm_layout),
        )

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipeline(trajectory)
