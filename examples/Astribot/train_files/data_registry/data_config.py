"""Astribot benchmark — DataConfig + LeRobotSingleDataset hook + mixtures.

Wires Astribot into starVLA's gr00t-style data path with the *minimum*
amount of glue:

  * declares a single ``state.cartesian`` / ``action.cartesian`` slice (raw
    34-dim ``cartesian_pose_*`` columns) — the slicing into per-arm chunks
    happens later, inside ``astribot_pipeline_np``;
  * declares ``video.head/right/left`` and the ``annotation.human.action.
    task_description`` channel for the prompt;
  * provides a ``make_dataset`` factory hook that returns a tiny
    ``LeRobotSingleDataset`` subclass which:
      - resolves Astribot's ``task_index = [coarse, fine]`` to the coarse
        string (the upstream gr00t loader assumes a scalar and crashes on
        ``.item()`` of a 2-element array),
      - applies ``AstribotPipeline`` to the raw 34-dim trajectories so the
        downstream ``QwenPIKM`` keeps seeing the (T, 14) delta+normalized
        dual-arm space it was designed for, and
      - resizes images to ``data_cfg.obs_image_size`` (default 240×320)
        rather than the gr00t base class's hard-coded 224×224.

All static training knobs (loss / lr / mixture) live in the YAML; this
file is purely about plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform

from examples.Astribot.astribot_transforms import (
    ROBOT_ID_MAPPING,
    astribot_pipeline_np,
    load_stats_for_robot,
)


# ---------------------------------------------------------------------------
# LeRobotSingleDataset subclass: resolves task_index pairs + applies pipeline.
# ---------------------------------------------------------------------------
class AstribotLeRobotDataset(LeRobotSingleDataset):
    """Astribot-aware subclass of ``LeRobotSingleDataset``.

    Overrides only what's needed:
      * ``get_language``: Astribot stores ``task_index`` as ``[coarse, fine]``
        — gr00t's default ``.item()`` would crash. We pick the coarse index.
      * ``_pack_sample``: take the raw 34-dim action/state, route them
        through ``astribot_pipeline_np`` (delta + normalize + dual-arm
        14-dim layout), and emit the starVLA contract.
    """

    # Will be set by ``make_dataset`` below.
    _astribot_stats = None
    _astribot_image_size: tuple[int, int] = (240, 320)   # (H, W)

    def get_language(self, trajectory_id: int, key: str, base_index: int) -> list[str]:
        # Re-implementation of the parent's path that handles Astribot's
        # ``task_index = [coarse, fine]`` ndarray instead of a scalar. The
        # rest of the logic (step_indices clamping, tasks.loc[...] lookup)
        # is unchanged.
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)

        assert key.startswith("annotation."), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None and subkey in annotation_meta
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key or key

        task_indices: list[int] = []
        for i in range(len(step_indices)):
            value = self.curr_traj_data[original_key].iloc[step_indices[i]]
            if isinstance(value, (int, float)):
                idx = int(value)
            else:
                arr = np.asarray(value).flatten()
                idx = int(arr[0])  # Astribot: coarse task index
            task_indices.append(idx)

        return self.tasks.loc[task_indices]["task"].tolist()

    def _pack_sample(self, data: dict) -> dict:
        # 1. Decode images and resize to the project's training resolution.
        H, W = self._astribot_image_size
        images: list[Image.Image] = []
        for video_key in self.modality_keys["video"]:
            frame = data[video_key][0]
            img = Image.fromarray(frame)
            if img.size != (W, H):
                img = img.resize((W, H), Image.BILINEAR)
            images.append(img)

        language = data[self.modality_keys["language"][0]][0]

        # 2. Raw 34-dim action / state come from the single ``cartesian`` key.
        # The modality.json declares ``state.cartesian`` (1, 34) and
        # ``action.cartesian`` (T, 34); we sanity-check the shape so any
        # future mis-configuration fails loudly here rather than silently
        # downstream.
        raw_action_34 = np.asarray(data[self.modality_keys["action"][0]], dtype=np.float32)
        raw_state_34 = np.asarray(data[self.modality_keys["state"][0]], dtype=np.float32)
        if raw_action_34.shape[-1] != 34 or raw_state_34.shape[-1] != 34:
            raise ValueError(
                f"Astribot expects raw 34-dim cartesian; got action={raw_action_34.shape}, "
                f"state={raw_state_34.shape}"
            )

        if self._astribot_stats is None:
            raise RuntimeError("AstribotLeRobotDataset._astribot_stats was not initialised")
        pipe = astribot_pipeline_np(raw_action_34, raw_state_34, self._astribot_stats)

        # 3. Pack into the starVLA framework contract.
        sample: dict[str, Any] = {
            "image": images,
            "lang": language,
            "action": pipe["action"],   # (T, 14) float32
            "state": pipe["state"],     # (1, 14) float32 — first row only
            "raw_state_34": raw_state_34[0].copy(),
            "robot_type": "S1-stationary",
            "robot_id": ROBOT_ID_MAPPING.get("S1-stationary", 0),
            "dataset_name": getattr(self, "_astribot_dataset_name", ""),
            "robot_tag": getattr(self, "tag", "astribot"),
        }
        return sample


# ---------------------------------------------------------------------------
# DataConfig — what ``make_LeRobotSingleDataset`` reads.
# ---------------------------------------------------------------------------
class AstribotS1DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.head", "video.right", "video.left"]
    state_keys = ["state.cartesian"]
    action_keys = ["action.cartesian"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]
    # Default: 32-frame action chunk. Trainer YAML can pin it via
    # ``framework.action_model.action_horizon`` / ``datasets.vla_data.action_horizon``;
    # ``make_dataset`` below threads ``data_cfg.action_horizon`` through.
    action_indices = list(range(32))

    def modality_config(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig

        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # No in-loader normalization: ``AstribotLeRobotDataset._pack_sample``
        # invokes the AstribotPipeline directly.
        return ComposedModalityTransform(transforms=[])

    def make_dataset(
        self,
        *,
        dataset_path: Path,
        modality_configs,
        transforms,
        embodiment_tag,
        video_backend: str,
        delete_pause_frame: bool = False,
        data_cfg=None,
        dataset_name: str | None = None,
    ) -> LeRobotSingleDataset:
        # Honour caller-provided action_horizon (defaults to 32 to match
        # this DataConfig). The caller (``train_starvla``) just hands us the
        # global ``vla_data`` cfg block.
        horizon = int((data_cfg or {}).get("action_horizon", 32)) if data_cfg else 32
        modality_configs = dict(modality_configs)
        modality_configs["action"].delta_indices = list(range(horizon))

        stats_path = (
            (data_cfg or {}).get(
                "stats_path",
                "./clap/assets/dataset_statistics_32.json",
            )
            if data_cfg
            else "./clap/assets/dataset_statistics_32.json"
        )
        image_size = (
            tuple(int(x) for x in (data_cfg or {}).get("obs_image_size", [240, 320]))
            if data_cfg
            else (240, 320)
        )

        ds = AstribotLeRobotDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )
        # Inject Astribot-specific runtime state (stats / image size /
        # dataset name) without touching the parent constructor signature.
        ds._astribot_stats = load_stats_for_robot(stats_path, "S1-stationary")
        ds._astribot_image_size = image_size
        ds._astribot_dataset_name = dataset_name or Path(dataset_path).name
        return ds


ROBOT_TYPE_CONFIG_MAP = {
    "astribot_s1": AstribotS1DataConfig(),
}


# ---------------------------------------------------------------------------
# Mixtures. Add new entries here when the team ships fresh Astribot corpora.
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "astribot_fold_clothes": [
        # path is relative to ``data_root_dir`` set in the YAML.
        ("init", 1.0, "astribot_s1"),
    ],
    "astribot_pretrain_pnp_daxiong": [
        ("0827_pretrain_pnp_daxiong", 1.0, "astribot_s1"),
    ],
}
