"""End-to-end smoke test against the bundled dummy LeRobot dataset.

Run with::

    python scripts/smoke_test_e2e.py

Verifies, against the 2-episode dummy dataset bundled in
``assets/dummy_dataset/`` (no external dataset access required):
  1. The pinned lerobot (bed90e3a) loads a v2.0 LeRobotDataset metadata.
  2. The clap CustomLeRobotDataset wrapper resolves the per-row task /
     subtask, robot type, and decodes all three video views.
  3. The MultiCustomLeRobotDataset iterator yields per-sample dicts.
  4. The LightningLerobot DataModule produces a non-empty collated batch.

If you have an external LIBERO dataset on disk and want to smoke against
that instead (covers v2.1 episodes_stats.jsonl), set::

    SMOKE_DATASET_PATH=/path/to/libero_spatial_no_noops_1.0.0_lerobot
"""
from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Default to the bundled dummy dataset; allow override via env var.
DEFAULT_DATASET = REPO_ROOT / "assets/dummy_dataset"
DATASET_PATH = Path(os.environ.get("SMOKE_DATASET_PATH", str(DEFAULT_DATASET)))
if not DATASET_PATH.exists():
    raise SystemExit(
        f"Dataset not found at {DATASET_PATH}. Set SMOKE_DATASET_PATH or "
        f"check assets/dummy_dataset/ exists in the repo."
    )
LIBERO_PATH = str(DATASET_PATH)


@contextmanager
def section(label: str):
    print(f"\n=== {label} ===", flush=True)
    try:
        yield
        print(f"[OK] {label}", flush=True)
    except Exception:
        print(f"[FAIL] {label}", flush=True)
        traceback.print_exc()
        raise


# ----------------------------------------------------------------------
with section("1) lerobot LeRobotDataset metadata — bare metal"):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
    print(f"  lerobot CODEBASE_VERSION (current): {CODEBASE_VERSION}")
    ds = LeRobotDataset(
        repo_id=DATASET_PATH.name,
        root=LIBERO_PATH,
    )
    print(f"  ✓ {ds.num_episodes} episodes, {ds.num_frames} frames")
    print(f"  ✓ codebase_version: {ds.meta._version}")
    print(f"  ✓ camera keys: {ds.meta.video_keys}")
    print(f"  ✓ feature keys (first 6): {list(ds.features.keys())[:6]}")
    # Skip the bare LeRobotDataset[0] test — Astribot uses 2-D task_index
    # which the upstream __getitem__ cannot scalarize. The clap wrapper
    # in test 2 handles this case.


# ----------------------------------------------------------------------
with section("2) clap CustomLeRobotDataset (v2.0/v2.1 compat)"):
    from clap.custom_lerobot import CustomLeRobotDataset
    cds = CustomLeRobotDataset(
        repo_id=DATASET_PATH.name,
        root=LIBERO_PATH,
        local_files_only=True,           # legacy kwarg — must be silently absorbed
        episodes=[0],                     # just episode 0 for speed
        # request the 3 camera views from the Astribot dummy dataset
        # (key set is robust because we read from meta).
        delta_timestamps={k: [0.0] for k in ds.meta.video_keys},
    )
    print(f"  ✓ CustomLeRobotDataset constructed; {len(cds)} samples")
    item = cds[0]
    print(f"  ✓ item keys (first 8): {sorted(item.keys())[:8]}, len={len(item)}")
    assert "task" in item, "expected a `task` field added by CustomLeRobotDataset"
    assert "robot_type" in item, "expected `robot_type` field added by CustomLeRobotDataset"
    print(f"  ✓ task: {item['task']!r}")
    print(f"  ✓ robot_type: {item['robot_type']}, robot_id: {item['robot_id']}")
    # At least one camera view should be decoded
    img_keys = [k for k in item if k.startswith("images_dict") or k.startswith("observation.images")]
    img_keys = [k for k in img_keys if not k.endswith("_is_pad")]
    assert img_keys, f"no image tensors in sample; keys={sorted(item.keys())}"
    print(f"  ✓ image keys: {img_keys}")
    print(f"  ✓ {img_keys[0]}: shape={tuple(item[img_keys[0]].shape)}, "
          f"dtype={item[img_keys[0]].dtype}")


# ----------------------------------------------------------------------
with section("3) clap MultiCustomLeRobotDataset — real iteration"):
    # MultiCustomLeRobotDataset filters fields via a hard-coded `kept_keys`
    # whitelist (`cartesian_so3_dict.*`, `images_dict.head.rgb`, etc.) that
    # matches the Astribot/AgiBot training format. We just verify the
    # iterator runs without crashing and returns the metadata fields.
    from clap.custom_lerobot import MultiCustomLeRobotDataset
    multi = MultiCustomLeRobotDataset(
        repo_ids=[LIBERO_PATH],
        delta_timestamps={"action": [0.0]},
        tolerances_s={LIBERO_PATH: 1e-3},
        robot_type_sampling_probs=None,
        parallel_load=False,
        num_samples_per_epoch=4,
    )
    pulled = 0
    for sample in multi:
        pulled += 1
        if pulled == 1:
            print(f"  ✓ first-iter sample keys: {sorted(sample.keys())[:8]}")
            print(f"  ✓ task: {sample['task']!r}")
            print(f"  ✓ robot_type: {sample['robot_type']}")
        if pulled >= 4:
            break
    assert pulled == 4, f"expected 4 samples, got {pulled}"
    print(f"  ✓ pulled {pulled} samples through the iterable wrapper")


# ----------------------------------------------------------------------
with section("4) clap LightningLerobot DataModule wraps the multi-dataset"):
    # Same `kept_keys` caveat as test 3 — we just confirm the collate
    # path produces a non-empty batch dict via DataLoader.
    from clap.dataset_lerobot import LightningLerobot
    dm = LightningLerobot(
        data_root=str(DATASET_PATH.parent),
        data_mix=[DATASET_PATH.name],
        batch_size=2,
        num_workers=0,
        delta_timestamps={"action": [0.0]},
        test_repo_id=LIBERO_PATH,
        test_episode=0,
    )
    dm.setup("fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    print(f"  ✓ batch keys: {sorted(batch.keys())[:8]}")
    print(f"  ✓ batch['robot_id']: {batch['robot_id'].tolist()}")
    print(f"  ✓ batch['task'][0]: {batch['task'][0][:60]!r}…")
    assert len(batch) > 0, "empty batch"


print(f"\n[ALL GREEN] e2e smoke test passed against {DATASET_PATH.name}.")
