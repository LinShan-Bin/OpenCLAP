"""One-shot setup: emit ``meta/modality.json`` for each Astribot LeRobot
v2.0 dataset directory.

starVLA's gr00t-style ``LeRobotSingleDataset`` requires this file. Astribot
pretrain dirs ship without it, so we generate one that:

  * declares the 34-dim ``cartesian_pose_command`` / ``cartesian_pose_state``
    columns as a single ``state.cartesian`` / ``action.cartesian`` slice,
  * remaps ``images_dict.{head,right,left}.rgb`` → ``video.head/right/left``,
  * exposes ``task_index`` as ``annotation.human.action.task_description``.

Usage::

    python examples/Astribot/train_files/setup_astribot_meta.py \\
        ./data/astribot_pretrain/0827_pretrain_pnp_daxiong \\
        ./data/fold_clothes/init

If a directory already contains ``meta/modality.json`` we leave it alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


# Same key mapping as clap/data_transform.py.
_VIDEO_KEY_MAP = {
    "head": "images_dict.head.rgb",
    "right": "images_dict.right.rgb",
    "left": "images_dict.left.rgb",
}


def _build_modality_dict(info: dict) -> dict:
    """Build the modality.json payload for one dataset, given its info.json."""
    feats = info.get("features", {})
    # Only emit video keys that actually exist in the corpus.
    video_block = {
        new_key: {"original_key": orig_key}
        for new_key, orig_key in _VIDEO_KEY_MAP.items()
        if orig_key in feats
    }
    if not video_block:
        raise RuntimeError(
            f"info.json has no Astribot images_dict.{'{head,right,left}'}.rgb keys; "
            f"available features: {sorted(feats)}"
        )

    return {
        "state": {
            "cartesian": {
                "start": 0,
                "end": 34,
                "original_key": "cartesian_so3_dict.cartesian_pose_state",
                "absolute": True,
                "dtype": "float64",
            }
        },
        "action": {
            "cartesian": {
                "start": 0,
                "end": 34,
                "original_key": "cartesian_so3_dict.cartesian_pose_command",
                "absolute": True,
                "dtype": "float64",
            }
        },
        "video": video_block,
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }


def _write_modality_json(dataset_dir: Path, *, force: bool) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    out_path = dataset_dir / "meta" / "modality.json"
    if not info_path.exists():
        print(f"[skip] {dataset_dir}: meta/info.json missing")
        return
    if out_path.exists() and not force:
        print(f"[ok]   {out_path} already exists")
        return
    with open(info_path, "r") as f:
        info = json.load(f)
    payload = _build_modality_dict(info)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[wrote] {out_path}")


def setup(dataset_dirs: Iterable[Path], *, force: bool = False) -> None:
    for d in dataset_dirs:
        _write_modality_json(Path(d), force=force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dirs", nargs="+", type=Path,
        help="One or more Astribot LeRobot v2.0 dataset directories.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing meta/modality.json.",
    )
    args = parser.parse_args()
    setup(args.dataset_dirs, force=args.force)


if __name__ == "__main__":
    main()
