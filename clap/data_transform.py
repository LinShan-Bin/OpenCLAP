import json
from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from scipy.signal import firwin

from clap.unified_action import ensure_arm_mask
from pathlib import Path as _Path
_CLAP_ROOT = _Path(__file__).resolve().parent
def _resolve(p):
    p = str(p)
    return p if _Path(p).is_absolute() else str(_CLAP_ROOT / p.lstrip('./'))




ASTRIBOT_KEY_MAPPING = {
    "cartesian_so3_dict.cartesian_pose_command": "action",
    "cartesian_so3_dict.cartesian_pose_state": "state",
    "images_dict.head.rgb": "observation.head",
    "images_dict.right.rgb": "observation.right",
    "images_dict.left.rgb": "observation.left",
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
        self.mapping = ASTRIBOT_KEY_MAPPING

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        ret_dict = {}
        for k, v in trajectory.items():
            if k in self.mapping:
                new_key = self.mapping[k]
            else:
                new_key = k
            ret_dict[new_key] = v
        return ret_dict


class ArmMask(nn.Module):
    def __init__(self, default_layout: str = "dual"):
        super(ArmMask, self).__init__()
        self.default_layout = default_layout

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        return ensure_arm_mask(trajectory, default_layout=self.default_layout)


class ActionNormalization(nn.Module):
    def __init__(self):
        super(ActionNormalization, self).__init__()
        dataset_stats = json.load(open(_resolve("assets/dataset_statistics_32.json")))
        astribot_stats = dataset_stats["S1-stationary"]
        agibot_stats = dataset_stats["agibot-go1"]
        
        action_q99 = torch.stack([
            torch.tensor(astribot_stats["action"]["q99"], dtype=torch.float32),
            torch.tensor(agibot_stats["action"]["q99"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
        ])
        action_q01 = torch.stack([
            torch.tensor(astribot_stats["action"]["q01"], dtype=torch.float32),
            torch.tensor(agibot_stats["action"]["q01"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
        ])
        self.register_buffer("action_q99", action_q99)
        self.register_buffer("action_q01", action_q01)
        
        state_q99 = torch.stack([
            torch.tensor(astribot_stats["state"]["q99"], dtype=torch.float32),
            torch.tensor(agibot_stats["state"]["q99"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
        ])
        state_q01 = torch.stack([
            torch.tensor(astribot_stats["state"]["q01"], dtype=torch.float32),
            torch.tensor(agibot_stats["state"]["q01"], dtype=torch.float32),
            torch.ones(14, dtype=torch.float32),
        ])
        self.register_buffer("state_q99", state_q99)
        self.register_buffer("state_q01", state_q01)
        
        normalize_mask = torch.tensor(
            [True] * 6 + [False] + [True] * 6 + [False], dtype=torch.bool)
        self.register_buffer("normalize_mask", normalize_mask)
        
        print("Loaded action normalization statistics")
        print(f"    Astribot action q99: {action_q99}")
        print(f"    Astribot action q01: {action_q01}")
        print(f"    Agibot action q99: {action_q99}")
        print(f"    Agibot action q01: {action_q01}")
        print(f"    Astribot state q99: {state_q99}")
        print(f"    Astribot state q01: {state_q01}")
        print(f"    Agibot state q99: {state_q99}")
        print(f"    Agibot state q01: {state_q01}")

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        action = trajectory["action"]
        robot_id = trajectory["robot_id"]
        if not isinstance(robot_id, torch.Tensor):
            robot_id = torch.tensor(robot_id, dtype=torch.long, device=action.device)
        robot_id = robot_id.to(device=action.device, dtype=torch.long).view(-1)

        action_q99 = self.action_q99[robot_id][:, None]
        action_q01 = self.action_q01[robot_id][:, None]
        action_scale = action_q99 - action_q01
        action_scale = torch.where(action_scale.abs() < 1e-8, torch.ones_like(action_scale), action_scale)
        action_normalized = (action - action_q01) / action_scale * 2 - 1
        action[:, :, self.normalize_mask] = action_normalized[:, :, self.normalize_mask]
        human_mask = robot_id == 2
        if human_mask.any():
            action[human_mask] = 0
        trajectory["action"] = action
        
        state = trajectory["state"]
        state_q99 = self.state_q99[robot_id][:, None]
        state_q01 = self.state_q01[robot_id][:, None]
        state_scale = state_q99 - state_q01
        state_scale = torch.where(state_scale.abs() < 1e-8, torch.ones_like(state_scale), state_scale)
        state_normalized = (state - state_q01) / state_scale * 2 - 1
        state[:, :, self.normalize_mask] = state_normalized[:, :, self.normalize_mask]
        if human_mask.any():
            state[human_mask] = 0
        trajectory["state"] = state
        
        return trajectory


class ActionDelta(nn.Module):
    def __init__(self):
        super(ActionDelta, self).__init__()
    
    def calculate_delta(self, action, state):
        delta_position = action[..., :3] - state[..., :3]
        
        rotation_act = self.r6d_to_mat(action[..., 3:])
        rotation_stat = self.r6d_to_mat(state[..., 3:])
        
        # Calculate delta rotation as relative rotation: delta_R = R_act @ R_state^T
        delta_rotation_mat = torch.matmul(rotation_act, rotation_stat.transpose(-2, -1))
        
        # Convert rotation matrix to euler angles (delta_R_x, delta_R_y, delta_R_z)
        delta_rotation = self.mat_to_euler(delta_rotation_mat)
        
        return torch.cat([delta_position, delta_rotation], dim=-1)
        
    @staticmethod
    def r6d_to_mat(r6d):
        r1 = r6d[..., 0: 3]
        r2 = r6d[..., 3: 6]
        b1 = r1 / (r1.norm(dim=-1, keepdim=True) + 1e-8)
        proj = (b1 * r2).sum(dim=-1, keepdim=True) * b1
        u2 = r2 - proj
        b2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-2)
    
    @staticmethod
    def mat_to_euler(rotation_matrix):
        """
        Convert rotation matrix to euler angles (XYZ convention)
        Returns: (roll_x, pitch_y, yaw_z)
        """
        # Extract elements from rotation matrix
        # R = [[r00, r01, r02],
        #      [r10, r11, r12],
        #      [r20, r21, r22]]
        r00 = rotation_matrix[..., 0, 0]
        r10 = rotation_matrix[..., 1, 0]
        r20 = rotation_matrix[..., 2, 0]
        r21 = rotation_matrix[..., 2, 1]
        r22 = rotation_matrix[..., 2, 2]
        
        # Calculate euler angles using XYZ convention
        pitch_y = torch.asin(-r20.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        roll_x = torch.atan2(r21, r22)
        yaw_z = torch.atan2(r10, r00)
        
        # Stack as (roll_x, pitch_y, yaw_z)
        euler = torch.stack([roll_x, pitch_y, yaw_z], dim=-1)
        return euler
    
    def so3_to_euler(self, so3):
        xyz = so3[..., :3]
        rot_mat = self.r6d_to_mat(so3[..., 3:])
        euler = self.mat_to_euler(rot_mat)
        return torch.cat([xyz, euler], dim=-1)

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        action = trajectory["action"]
        state = trajectory["state"]
        
        left_action = action[..., 9: 18]
        left_gripper = action[..., 18: 19]
        right_action = action[..., 19: 28]
        right_gripper = action[..., 28: 29]
        
        left_state = state[..., 9: 18]
        left_gripper_state = state[..., 18: 19]
        right_state = state[..., 19: 28]
        right_gripper_state = state[..., 28: 29]
        
        delta_left = self.calculate_delta(left_action, left_state[..., :1, :])
        delta_right = self.calculate_delta(right_action, right_state[..., :1, :])
        
        left_state_euler = self.so3_to_euler(left_state)
        right_state_euler = self.so3_to_euler(right_state)
        
        # Before: 0 -> open, 100 -> closed
        # After: 1 -> closed, -1 -> open
        left_gripper = left_gripper / 100.0 * 2 - 1
        right_gripper = right_gripper / 100.0 * 2 - 1
        left_gripper = left_gripper.clamp(-1, 1)
        right_gripper = right_gripper.clamp(-1, 1)
        
        left_gripper_state = left_gripper_state / 100.0 * 2 - 1
        right_gripper_state = right_gripper_state / 100.0 * 2 - 1
        left_gripper_state = left_gripper_state.clamp(-1, 1)
        right_gripper_state = right_gripper_state.clamp(-1, 1)
        
        trajectory["action"] = torch.cat([
            delta_left, left_gripper, delta_right, right_gripper], dim=-1)
        trajectory["state"] = torch.cat([
            left_state_euler, left_gripper_state, right_state_euler, right_gripper_state], dim=-1)
        trajectory["action_static_reference"] = trajectory["action"].clone()
        
        return trajectory


class TorchZeroPhaseFIR(nn.Module):
    """
    PyTorch implementation of a zero-phase FIR filter using convolution.
    This is a parallel-friendly alternative to the recursive IIR filter.

    Args:
        fs (float): The sampling frequency of the signal.
        fc (float): The cutoff frequency of the low-pass filter.
        numtaps (int): The length of the FIR filter kernel (number of taps). 
                       This is equivalent to the filter 'order' + 1. 
                       A larger numtaps gives a sharper cutoff, but is computationally
                       more expensive and has more boundary effects. Must be an odd number
                       to easily create a 'same' convolution with integer padding.
    """
    def __init__(self, fs: float = 30, fc: float = 8, numtaps: int = 19):
        super(TorchZeroPhaseFIR, self).__init__()

        # Ensure numtaps is odd for 'same' padding calculation
        if numtaps % 2 == 0:
            raise ValueError("numtaps must be an odd number.")
            
        # 1. Design FIR filter kernel using scipy.signal.firwin
        # This gives us the 'b' coefficients of the FIR filter.
        # The window function ('hamming') helps to reduce ripples in the stopband.
        kernel_np = firwin(numtaps=numtaps, cutoff=fc, fs=fs, window='hamming', pass_zero='lowpass')
        
        # 2. Register the kernel as a buffer.
        # The kernel needs to have a shape of [out_channels, in_channels/groups, kernel_size]
        # For our use case: [1, 1, numtaps]
        self.register_buffer('kernel', torch.from_numpy(kernel_np.copy()).float().view(1, 1, -1))
        
        # Padding required to keep the output length the same as input length
        self.padding = (numtaps - 1) // 2
        print(f"TorchZeroPhaseFIR initialized: fs={fs}, fc={fc}, numtaps={numtaps}. Padding: {self.padding}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply zero-phase FIR filtering to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D] (Batch, Time, Dimension).

        Returns:
            torch.Tensor: Filtered tensor of the same shape.
        """
        dtype = x.dtype
        
        # Ensure input is at least 2D
        if x.dim() < 2:
            raise ValueError("Input tensor must have at least 2 dimensions [T, D] or [B, T, D]")
        
        # Add a batch dimension if input is [T, D]
        is_single_sequence = x.dim() == 2
        if is_single_sequence:
            x = x.unsqueeze(0)

        # Get original shape: Batch, Time, Dimension
        B, T, D = x.shape
        
        # `conv1d` expects input of shape [Batch, Channels, Length].
        # We treat the 'D' dimension as channels.
        # So, we permute from [B, T, D] to [B, D, T]
        x_permuted = x.permute(0, 2, 1)

        # To apply the same filter to each of the D channels independently,
        # we reshape the input and use grouped convolution.
        # Reshape from [B, D, T] to [B*D, 1, T]
        x_reshaped = x_permuted.reshape(B * D, 1, T)

        # The kernel is of shape [1, 1, numtaps]. `conv1d` will apply it.
        # We use padding to ensure the output length is the same as the input length.
        
        # 1. Forward pass convolution
        y_forward = F.conv1d(x_reshaped, self.kernel, padding=self.padding)
        
        # 2. Reverse the filtered signal along the time dimension
        y_reversed = torch.flip(y_forward, dims=[2])
        
        # 3. Backward pass convolution
        y_backward = F.conv1d(y_reversed, self.kernel, padding=self.padding)
        
        # 4. Reverse the signal again to get the final result
        y_final_reshaped = torch.flip(y_backward, dims=[2])
        
        # Reshape back to [B, D, T]
        y_final_permuted = y_final_reshaped.reshape(B, D, T)
        
        # Permute back to the original [B, T, D] format
        y_final = y_final_permuted.permute(0, 2, 1)

        # Remove the batch dimension if the original input was [T, D]
        if is_single_sequence:
            y_final = y_final.squeeze(0)
            
        return y_final.to(dtype)


class ActionFilter(nn.Module):
    def __init__(self, fs: float = 30, fc: float = 8, numtaps: int = 19):
        super(ActionFilter, self).__init__()
        self.filter = TorchZeroPhaseFIR()
        
        filter_mask = torch.tensor(
            [True] * 6 + [False] + [True] * 6 + [False], dtype=torch.bool)
        self.register_buffer("filter_mask", filter_mask)
        
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        T = trajectory["action"].shape[1]
        
        action = trajectory["action"]
        action[..., self.filter_mask] = self.filter(action[..., self.filter_mask])
        trajectory["action"] = action[..., T // 3: 2 * T // 3, :]

        return trajectory


class ActionDenormalization(nn.Module):
    """
    Denormalize actions back to original scale.
    This is the inverse operation of ActionNormalization.

    Supports xyz_rpy (14-dim) action space.

    Normalization formula: normalized = (action - q01) / (q99 - q01) * 2 - 1
    Denormalization formula: action = (normalized + 1) / 2 * (q99 - q01) + q01
    """
    def __init__(self, robot_id: int = 0, action_space: str = "xyz_rpy"):
        super(ActionDenormalization, self).__init__()
        assert action_space == "xyz_rpy", f"Only xyz_rpy is supported, got {action_space}"

        self.action_space = action_space
        self.robot_id = robot_id
        extra_action_q99 = None
        extra_action_q01 = None
        extra_state_q99 = None
        extra_state_q01 = None

        # Load xyz_rpy statistics (14-dim)
        dataset_stats = json.load(open(_resolve("assets/dataset_statistics_32.json")))
        base_robot_stats = [
            dataset_stats["S1-stationary"],
            dataset_stats["agibot-go1"],
            None,  # human placeholder
        ]
        extra_robot_stats = [
            dataset_stats["franka"],
            dataset_stats["fractal"],
            dataset_stats["widowx"],
            dataset_stats.get("droid", dataset_stats["franka"]),
        ]

        def make_stats_tensor(robot_stats: list[dict | None], kind: str, quantile: str) -> torch.Tensor:
            values = []
            for stats in robot_stats:
                if stats is None:
                    values.append(torch.ones(14, dtype=torch.float32))
                else:
                    values.append(torch.tensor(stats[kind][quantile], dtype=torch.float32))
            return torch.stack(values)

        action_q99 = make_stats_tensor(base_robot_stats, "action", "q99")
        action_q01 = make_stats_tensor(base_robot_stats, "action", "q01")
        state_q99 = make_stats_tensor(base_robot_stats, "state", "q99")
        state_q01 = make_stats_tensor(base_robot_stats, "state", "q01")
        extra_action_q99 = make_stats_tensor(extra_robot_stats, "action", "q99")
        extra_action_q01 = make_stats_tensor(extra_robot_stats, "action", "q01")
        extra_state_q99 = make_stats_tensor(extra_robot_stats, "state", "q99")
        extra_state_q01 = make_stats_tensor(extra_robot_stats, "state", "q01")

        normalize_mask = torch.tensor(
            [True] * 6 + [False] + [True] * 6 + [False], dtype=torch.bool)
        
        self.register_buffer("action_q99", action_q99)
        self.register_buffer("action_q01", action_q01)
        self.register_buffer("state_q99", state_q99)
        self.register_buffer("state_q01", state_q01)
        if extra_action_q99 is not None:
            self.register_buffer("extra_action_q99", extra_action_q99)
            self.register_buffer("extra_action_q01", extra_action_q01)
            self.register_buffer("extra_state_q99", extra_state_q99)
            self.register_buffer("extra_state_q01", extra_state_q01)
        self.register_buffer("normalize_mask", normalize_mask)
        
        print(f"Loaded action denormalization statistics: action_space={action_space}, robot_id={robot_id}")

    def _select_stats(self, prefix: str, robot_id: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(robot_id, torch.Tensor):
            robot_id = int(robot_id.flatten()[0].item())
        else:
            robot_id = int(robot_id)

        q99 = getattr(self, f"{prefix}_q99")
        q01 = getattr(self, f"{prefix}_q01")
        if robot_id < q99.shape[0]:
            return q99[robot_id], q01[robot_id]

        extra_q99 = getattr(self, f"extra_{prefix}_q99", None)
        extra_q01 = getattr(self, f"extra_{prefix}_q01", None)
        extra_idx = robot_id - q99.shape[0]
        if extra_q99 is not None and 0 <= extra_idx < extra_q99.shape[0]:
            return extra_q99[extra_idx], extra_q01[extra_idx]

        raise IndexError(f"No denormalization stats for robot_id={robot_id}")
    
    def denormalize_action(self, action_normalized, robot_id=None):
        """
        Denormalize action tensor. Operates on 14-dim xyz_rpy actions.
        For single arm actions, please concatenate left and right arms first.
        
        Args:
            action_normalized: [..., D] tensor of normalized actions in range [-1, 1]
                             D = 14 for xyz_rpy
            robot_id: optional robot_id to use. If None, uses self.robot_id
        Returns:
            action: [..., D] tensor of denormalized actions
        """
        if robot_id is None:
            robot_id = self.robot_id
        
        action = action_normalized.clone()
        action_dim = action_normalized.shape[-1]
        
        # Validate dimension based on action_space
        expected_dim = 14
        if action_dim != expected_dim:
            raise ValueError(f"action_normalized must have dimension {expected_dim} for {self.action_space}, got {action_dim}. "
                           f"Please concatenate left and right arms first.")
        
        # Get statistics for this robot
        action_q99, action_q01 = self._select_stats("action", robot_id)
        mask = self.normalize_mask
        
        # Denormalize: action = (normalized + 1) / 2 * (q99 - q01) + q01
        action_denormalized = (action_normalized + 1) / 2 * (action_q99 - action_q01) + action_q01
        action[..., mask] = action_denormalized[..., mask]
        
        return action
    
    def denormalize_state(self, state_normalized, robot_id=None):
        """
        Denormalize state tensor. Operates on 14-dim xyz_rpy actions.
        For single arm states, please concatenate left and right arms first.
        
        Args:
            state_normalized: [..., D] tensor of normalized states in range [-1, 1]
                            D = 14 for xyz_rpy
            robot_id: optional robot_id to use. If None, uses self.robot_id
        Returns:
            state: [..., D] tensor of denormalized states
        """
        if robot_id is None:
            robot_id = self.robot_id
        
        state = state_normalized.clone()
        state_dim = state_normalized.shape[-1]
        
        # Validate dimension based on action_space
        expected_dim = 14
        if state_dim != expected_dim:
            raise ValueError(f"state_normalized must have dimension {expected_dim} for {self.action_space}, got {state_dim}. "
                           f"Please concatenate left and right arms first.")
        
        # Get statistics for this robot
        state_q99, state_q01 = self._select_stats("state", robot_id)
        mask = self.normalize_mask
        
        # Denormalize: state = (normalized + 1) / 2 * (q99 - q01) + q01
        state_denormalized = (state_normalized + 1) / 2 * (state_q99 - state_q01) + state_q01
        state[..., mask] = state_denormalized[..., mask]
        
        return state
    
    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Denormalize actions and states in a trajectory dict.
        """
        robot_id = trajectory.get("robot_id", self.robot_id)
        
        if "action" in trajectory:
            trajectory["action"] = self.denormalize_action(trajectory["action"], robot_id)
        
        if "state" in trajectory:
            trajectory["state"] = self.denormalize_state(trajectory["state"], robot_id)
        
        return trajectory


class DeltaToAbsolute(nn.Module):
    """
    Convert delta actions back to absolute actions.
    This is the inverse operation of ActionDelta.

    Supports xyz_rpy (7-dim per arm) action space.
    """
    def __init__(self, action_space: str = "xyz_rpy"):
        super(DeltaToAbsolute, self).__init__()
        assert action_space == "xyz_rpy", f"Only xyz_rpy is supported, got {action_space}"
        self.action_space = action_space
    
    @staticmethod
    def euler_to_mat(euler):
        """
        Convert euler angles (roll, pitch, yaw) to rotation matrix using XYZ convention
        Args:
            euler: [..., 3] tensor of euler angles (roll_x, pitch_y, yaw_z)
        Returns:
            rotation_matrix: [..., 3, 3] rotation matrix
        """
        roll_x = euler[..., 0]
        pitch_y = euler[..., 1]
        yaw_z = euler[..., 2]
        
        # Compute rotation matrices for each axis
        cos_roll = torch.cos(roll_x)
        sin_roll = torch.sin(roll_x)
        cos_pitch = torch.cos(pitch_y)
        sin_pitch = torch.sin(pitch_y)
        cos_yaw = torch.cos(yaw_z)
        sin_yaw = torch.sin(yaw_z)
        
        # Build rotation matrix using XYZ convention: R = Rz * Ry * Rx
        # First, create rotation matrices for each axis
        zeros = torch.zeros_like(roll_x)
        ones = torch.ones_like(roll_x)
        
        # R_x (roll)
        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_roll, -sin_roll], dim=-1),
            torch.stack([zeros, sin_roll, cos_roll], dim=-1)
        ], dim=-2)
        
        # R_y (pitch)
        Ry = torch.stack([
            torch.stack([cos_pitch, zeros, sin_pitch], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_pitch, zeros, cos_pitch], dim=-1)
        ], dim=-2)
        
        # R_z (yaw)
        Rz = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)
        
        # Combine: R = Rz @ Ry @ Rx
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))
        return R
    
    @staticmethod
    def mat_to_euler(rotation_matrix):
        """
        Convert rotation matrix to euler angles (XYZ convention)
        Args:
            rotation_matrix: [..., 3, 3] rotation matrix
        Returns:
            euler: [..., 3] tensor of euler angles (roll_x, pitch_y, yaw_z)
        """
        r00 = rotation_matrix[..., 0, 0]
        r10 = rotation_matrix[..., 1, 0]
        r20 = rotation_matrix[..., 2, 0]
        r21 = rotation_matrix[..., 2, 1]
        r22 = rotation_matrix[..., 2, 2]
        
        pitch_y = torch.asin(-r20.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        roll_x = torch.atan2(r21, r22)
        yaw_z = torch.atan2(r10, r00)
        
        euler = torch.stack([roll_x, pitch_y, yaw_z], dim=-1)
        return euler
    


    def convert(self, delta_action, state):
        """
        Convert delta action to absolute action.
        
        This reverses the operation in ActionDelta/ActionDeltaR6D:
        - Delta calculation: delta_pos = action_pos - state_pos
        - Inverse: action_pos = delta_pos + state_pos
        - Delta calculation: delta_R = R_action @ R_state^T
        - Inverse: R_action = delta_R @ R_state
        
        Args:
            delta_action: [..., 7] tensor of delta actions (delta_xyz, delta_euler, gripper)
            state: [..., 7] tensor of state (xyz, euler, gripper)
        Returns:
            absolute_action: [..., 7] tensor of absolute actions
        """
        # xyz_rpy: 7-dim per arm (xyz + euler + gripper)
        delta_pos = delta_action[..., :3]
        delta_euler = delta_action[..., 3:6]
        delta_gripper = delta_action[..., 6:7]

        state_pos = state[..., :3]
        state_euler = state[..., 3:6]

        # Absolute position: state_pos + delta_pos
        abs_pos = state_pos + delta_pos

        # Absolute rotation: delta_R @ state_R
        state_R = self.euler_to_mat(state_euler)
        delta_R = self.euler_to_mat(delta_euler)
        abs_R = torch.matmul(delta_R, state_R)
        abs_euler = self.mat_to_euler(abs_R)

        # Combine
        absolute_action = torch.cat([abs_pos, abs_euler, delta_gripper], dim=-1)
        return absolute_action


class AstribotPipeline(nn.Module):
    def __init__(self, fc: float = 8):
        super(AstribotPipeline, self).__init__()
        
        self.pipeline = nn.Sequential(
            KeyMapping(),
            ActionDelta(),
            ActionNormalization(),
            ArmMask(default_layout="dual"),
        )

    def forward(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        trajectory = self.pipeline(trajectory)
        return trajectory

