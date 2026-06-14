from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor


ACT_PAD_TOKEN = "<ACT_PAD>"

ARM_LAYOUT_TO_MASK = {
    "dual": (1, 1),
    "single_left": (1, 0),
    "single_right": (0, 1),
}


def normalize_arm_layout(arm_layout: Union[str, bytes]) -> str:
    if isinstance(arm_layout, bytes):
        arm_layout = arm_layout.decode("utf-8")
    if not isinstance(arm_layout, str):
        raise TypeError(f"arm_layout must be a string, got {type(arm_layout)}")
    arm_layout = arm_layout.lower()
    aliases = {
        "left": "single_left",
        "right": "single_right",
        "single": "single_right",
        "bimanual": "dual",
    }
    arm_layout = aliases.get(arm_layout, arm_layout)
    if arm_layout not in ARM_LAYOUT_TO_MASK:
        raise ValueError(f"Unsupported arm_layout: {arm_layout}")
    return arm_layout


def expand_arm_layouts(
    arm_layout: Optional[Union[str, Sequence[str]]],
    batch_size: int,
    default_layout: str = "dual",
) -> List[str]:
    if arm_layout is None:
        arm_layout = default_layout
    if isinstance(arm_layout, str):
        return [normalize_arm_layout(arm_layout)] * batch_size
    layouts = [normalize_arm_layout(layout) for layout in arm_layout]
    if len(layouts) != batch_size:
        raise ValueError(f"arm_layout length {len(layouts)} does not match batch size {batch_size}")
    return layouts


def make_arm_mask(
    arm_layout: Optional[Union[str, Sequence[str]]],
    batch_size: int,
    *,
    default_layout: str = "dual",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    layouts = expand_arm_layouts(arm_layout, batch_size, default_layout=default_layout)
    return torch.tensor([ARM_LAYOUT_TO_MASK[layout] for layout in layouts], device=device, dtype=dtype)


def get_arm_mask(
    batch: Dict,
    batch_size: int,
    *,
    default_layout: str = "dual",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    if "arm_mask" in batch:
        arm_mask = batch["arm_mask"]
        if not isinstance(arm_mask, Tensor):
            arm_mask = torch.tensor(arm_mask)
        arm_mask = arm_mask.to(device=device, dtype=dtype)
        if arm_mask.ndim == 1:
            arm_mask = arm_mask.view(1, 2).expand(batch_size, -1)
        if arm_mask.shape != (batch_size, 2):
            raise ValueError(f"arm_mask must have shape [{batch_size}, 2], got {tuple(arm_mask.shape)}")
        return arm_mask
    return make_arm_mask(
        batch.get("arm_layout", default_layout),
        batch_size,
        default_layout=default_layout,
        device=device,
        dtype=dtype,
    )


def ensure_arm_mask(
    trajectory: Dict,
    *,
    default_layout: str = "dual",
    dtype: torch.dtype = torch.float32,
) -> Dict:
    action = trajectory.get("action")
    state = trajectory.get("state")
    ref = action if isinstance(action, Tensor) else state
    if ref is None:
        raise KeyError("trajectory must contain action or state to infer batch size")
    batch_size = ref.shape[0]
    arm_mask = get_arm_mask(
        trajectory,
        batch_size,
        default_layout=default_layout,
        device=ref.device,
        dtype=dtype,
    )
    layouts = expand_arm_layouts(trajectory.get("arm_layout", default_layout), batch_size, default_layout)
    trajectory["arm_mask"] = arm_mask
    trajectory["arm_layout"] = layouts
    return trajectory


def split_arm_actions(action: Tensor, action_dim_per_arm: int) -> tuple[Tensor, Tensor]:
    expected_dim = action_dim_per_arm * 2
    if action.shape[-1] != expected_dim:
        raise ValueError(f"Expected last dim {expected_dim}, got {action.shape[-1]}")
    return action[..., :action_dim_per_arm], action[..., action_dim_per_arm:expected_dim]


def zero_inactive_slots(action_or_state: Tensor, arm_mask: Tensor, action_dim_per_arm: int) -> Tensor:
    left, right = split_arm_actions(action_or_state, action_dim_per_arm)
    arm_mask = arm_mask.to(device=action_or_state.device, dtype=action_or_state.dtype)
    if arm_mask.ndim == 1:
        arm_mask = arm_mask.view(1, 2).expand(action_or_state.shape[0], -1)
    view_shape = [arm_mask.shape[0]] + [1] * (left.ndim - 2) + [1]
    left = left * arm_mask[:, 0].view(view_shape)
    right = right * arm_mask[:, 1].view(view_shape)
    return torch.cat([left, right], dim=-1)


def flatten_active_arm_actions(action: Tensor, arm_mask: Optional[Tensor], action_dim_per_arm: int) -> Tensor:
    batch_size = action.shape[0]
    if arm_mask is None:
        arm_mask = torch.ones(batch_size, 2, device=action.device, dtype=torch.bool)
    else:
        arm_mask = arm_mask.to(device=action.device, dtype=torch.bool)
    left_action, right_action = split_arm_actions(action, action_dim_per_arm)

    # Preserve legacy dual-arm order exactly: all left sequences, then all right sequences.
    active_left = left_action[arm_mask[:, 0]]
    active_right = right_action[arm_mask[:, 1]]
    if active_left.numel() == 0 and active_right.numel() == 0:
        raise ValueError("arm_mask must activate at least one arm")
    return torch.cat([active_left, active_right], dim=0)


def scatter_active_arm_actions(active_action: Tensor, arm_mask: Tensor, action_dim_per_arm: int) -> Tensor:
    arm_mask = arm_mask.to(device=active_action.device, dtype=torch.bool)
    batch_size = arm_mask.shape[0]
    time_dim = active_action.shape[1]
    output = active_action.new_zeros(batch_size, time_dim, action_dim_per_arm * 2)
    n_left = int(arm_mask[:, 0].sum().item())
    n_right = int(arm_mask[:, 1].sum().item())
    if active_action.shape[0] != n_left + n_right:
        raise ValueError(
            f"active_action batch {active_action.shape[0]} does not match active arms {n_left + n_right}"
        )
    if n_left > 0:
        output[arm_mask[:, 0], :, :action_dim_per_arm] = active_action[:n_left]
    if n_right > 0:
        output[arm_mask[:, 1], :, action_dim_per_arm:] = active_action[n_left:n_left + n_right]
    return output


def expand_arm_mask_to_action_dim(arm_mask: Tensor, action_dim_per_arm: int, target: Tensor) -> Tensor:
    arm_mask = arm_mask.to(device=target.device, dtype=target.dtype)
    if arm_mask.ndim == 1:
        arm_mask = arm_mask.view(1, 2).expand(target.shape[0], -1)
    mask = torch.cat(
        [
            arm_mask[:, 0:1].expand(-1, action_dim_per_arm),
            arm_mask[:, 1:2].expand(-1, action_dim_per_arm),
        ],
        dim=-1,
    )
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)
    return mask.expand_as(target)


def format_action_token_strings(
    left_indices: Tensor,
    right_indices: Tensor,
    arm_mask: Tensor,
    *,
    pad_token: str = ACT_PAD_TOKEN,
) -> List[str]:
    left_indices = left_indices.detach().cpu()
    right_indices = right_indices.detach().cpu()
    arm_mask = arm_mask.detach().cpu().bool()
    if left_indices.shape != right_indices.shape:
        raise ValueError("left_indices and right_indices must have the same shape")
    if arm_mask.shape != (left_indices.shape[0], 2):
        raise ValueError(f"arm_mask shape {tuple(arm_mask.shape)} does not match batch size")

    action_tokens = []
    for i in range(left_indices.shape[0]):
        left_active = bool(arm_mask[i, 0])
        right_active = bool(arm_mask[i, 1])
        if left_active and right_active:
            # Legacy Stage 3 format: right arm tokens first, then left arm tokens.
            tokens = right_indices[i].tolist() + left_indices[i].tolist()
            action_tokens.append("".join([f"<ACT_{idx}>" for idx in tokens]))
        elif right_active:
            tokens = right_indices[i].tolist()
            action_tokens.append("".join([f"<ACT_{idx}>" for idx in tokens]) + pad_token)
        elif left_active:
            tokens = left_indices[i].tolist()
            action_tokens.append("".join([f"<ACT_{idx}>" for idx in tokens]) + pad_token)
        else:
            raise ValueError("arm_mask must activate at least one arm")
    return action_tokens


def _pad_or_trim_indices(indices: Sequence[int], length: int, pad_value: int = 0) -> List[int]:
    indices = list(indices[:length])
    if len(indices) < length:
        indices.extend([pad_value] * (length - len(indices)))
    return indices


def split_action_token_indices(
    clap_indices: Sequence[int],
    num_action_codes: int,
    *,
    arm_layout: str = "dual",
    pad_value: int = 0,
) -> Dict[str, Optional[List[int]]]:
    arm_layout = normalize_arm_layout(arm_layout)
    clap_indices = list(clap_indices)
    if len(clap_indices) >= num_action_codes * 2:
        right_indices = _pad_or_trim_indices(clap_indices[:num_action_codes], num_action_codes, pad_value)
        left_indices = _pad_or_trim_indices(
            clap_indices[num_action_codes:num_action_codes * 2],
            num_action_codes,
            pad_value,
        )
        return {
            "left_indices": left_indices,
            "right_indices": right_indices,
            "arm_mask": [1, 1],
            "arm_layout": "dual",
        }

    single_indices = _pad_or_trim_indices(clap_indices[:num_action_codes], num_action_codes, pad_value)
    if arm_layout == "single_left":
        return {
            "left_indices": single_indices,
            "right_indices": None,
            "arm_mask": [1, 0],
            "arm_layout": "single_left",
        }
    return {
        "left_indices": None,
        "right_indices": single_indices,
        "arm_mask": [0, 1],
        "arm_layout": "single_right",
    }


def decode_action_tokens_to_numpy(
    clap,
    clap_indices: Sequence[int],
    *,
    num_action_codes: int,
    action_dim_per_arm: int,
    chunk_size: int,
    arm_layout: str = "dual",
) -> np.ndarray:
    parsed = split_action_token_indices(
        clap_indices,
        num_action_codes,
        arm_layout=arm_layout,
    )
    action = np.zeros((chunk_size, action_dim_per_arm * 2), dtype=np.float32)

    device = getattr(clap, "device", None)
    if device is None:
        device = next(clap.parameters()).device

    def decode_one_arm(indices: Sequence[int]) -> np.ndarray:
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        latent = clap.vq_t.codebook(indices_tensor)
        z_q_decode = latent.unsqueeze(1)
        decoded = clap.action_vae.decode(z_q_decode, [chunk_size])
        return decoded[0].detach().cpu().float().numpy()

    if parsed["left_indices"] is not None and parsed["right_indices"] is not None:
        left_indices = torch.tensor(parsed["left_indices"], device=device, dtype=torch.long)
        right_indices = torch.tensor(parsed["right_indices"], device=device, dtype=torch.long)
        left_latent = clap.vq_t.codebook(left_indices)
        right_latent = clap.vq_t.codebook(right_indices)
        combined_latent = torch.stack([left_latent, right_latent], dim=0)
        z_q_decode = combined_latent.permute(1, 0, 2)
        decoded = clap.action_vae.decode(z_q_decode, [chunk_size, chunk_size])
        action[:, :action_dim_per_arm] = decoded[0].detach().cpu().float().numpy()
        action[:, action_dim_per_arm:] = decoded[1].detach().cpu().float().numpy()
        return action

    if parsed["left_indices"] is not None:
        action[:, :action_dim_per_arm] = decode_one_arm(parsed["left_indices"])
    if parsed["right_indices"] is not None:
        action[:, action_dim_per_arm:] = decode_one_arm(parsed["right_indices"])
    return action
