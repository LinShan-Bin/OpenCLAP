# Astribot · QwenPIKM

End-to-end recipe for training and serving QwenPIKM on Astribot S1-stationary
LeRobot data, ported from `clap/model_clap_vla_fm.py` and
`reference/async/websocket_policy_asynchronous_astribot.py`.

## Layout

```
examples/Astribot/
  astribot_transforms.py             # numpy-only AstribotPipeline helpers
  train_files/
    setup_astribot_meta.py           # one-shot: write meta/modality.json per dataset
    starvla_astribot_qwenpikm.yaml   # base config (KM + KI knobs)
    run_astribot_qwenpikm.sh         # launcher; MODE=km | ki
    data_registry/
      data_config.py                 # AstribotS1DataConfig + make_dataset hook + mixtures
  eval_files/
    async_policy_server.py           # dual-channel ws server for QwenPIKM
    run_async_policy_server.sh       # launcher
    smoke_async_server.py            # end-to-end ws smoke test
    smoke_pipeline.py                # 3 audit tests: CLAP / pipeline / metadata
```

The Astribot adapter rides on top of starVLA's gr00t-style
`LeRobotSingleDataset`. The minimum-mod glue is:

1. `setup_astribot_meta.py` writes `meta/modality.json` in each Astribot
   dataset dir, declaring a single 34-dim `state.cartesian` /
   `action.cartesian` slice and remapping `images_dict.{head,right,left}.rgb`
   → `video.head/right/left`.
2. `data_registry/data_config.py` registers `astribot_s1` as a `robot_type`
   and provides a `make_dataset` factory that returns a tiny subclass of
   `LeRobotSingleDataset`. The subclass overrides only:
   * `get_language` — Astribot's `task_index` is `[coarse, fine]`, not a
     scalar; pick the coarse string.
   * `_pack_sample` — apply `astribot_pipeline_np` (delta + normalize +
     dual-arm 14-dim layout) and emit the starVLA contract:
     `{image: [PIL], lang: str, action[T, 14], state[1, 14], raw_state_34}`.

## Setup

```bash
# Generate meta/modality.json for each Astribot LeRobot v2.0 dataset dir.
# Idempotent — already-existing files are kept (use --force to overwrite).
python examples/Astribot/train_files/setup_astribot_meta.py \
    ./data/fold_clothes/init \
    ./data/astribot_pretrain/0827_pretrain_pnp_daxiong
```

To register a new mixture, add an entry to
`data_registry/data_config.py:DATASET_NAMED_MIXTURES`.

## Train

```bash
# KL knowledge-matching (default)
bash examples/Astribot/train_files/run_astribot_qwenpikm.sh

# KI (Knowledge Insulating): VLM frozen, action expert only.
# Mutually exclusive with KL — QwenPIKM enforces this in __init__.
MODE=ki bash examples/Astribot/train_files/run_astribot_qwenpikm.sh

# Tiny smoke run (3 steps, no checkpoint save)
SMOKE_TEST=1 bash examples/Astribot/train_files/run_astribot_qwenpikm.sh
```

KI under the hood:
- `framework.enable_ki=True` runs the VLM forward inside `torch.no_grad()` —
  no graph builds through it (memory-saving).
- `trainer.freeze_modules='qwen_vl_interface'` sets `requires_grad=False` so
  VLM params are excluded from `param_groups` by `build_param_lr_groups`.

The two together give the "Knowledge Insulating" recipe from CLAP-VLA-FM:
gradient-free VLM, action expert trains as usual.

## Serve (async websocket, dual-channel)

```bash
# Real serving (production checkpoint)
CKPT=./ckpts/Checkpoints/astribot_qwenpikm_km/checkpoints/steps_50000_pytorch_model \
PORT=8000 \
bash examples/Astribot/eval_files/run_async_policy_server.sh

# Smoke test (build framework from YAML, no real weights):
python examples/Astribot/eval_files/smoke_async_server.py
```

The server protocol matches `reference/async/README.md` exactly:

* Sender: connect → send `{"role": "sender"}` → repeatedly send obs frames or
  `{"reset": 1}`. Obs schema:
  ```python
  {
    "images": {"cam_high": [3, H, W] float32 in [0, 1] CHW,
               "cam_left_wrist": ..., "cam_right_wrist": ...},
    "state": [34] float64,
    "obs_timestamp": float,
    "prompt": str,    # optional; falls back to --default_prompt
  }
  ```
* Receiver: connect → send `{"role": "receiver"}` → first frame is a
  `{"kind": "metadata", "train_freq": 30, "action_chunk_size": 32,
  "action_dim": 34, ...}` handshake (RTG init), thereafter
  ```python
  {"actions": ndarray[chunk, 34], "obs_timestamp": float, ...}
  ```

The server publishes 29 dimensions of *absolute* robot command
(`[torso(9), left_arm_so3(9), left_grip(1), right_arm_so3(9), right_grip(1)]`)
plus the state's head/chassis tail to fill out 34 dims, exactly like the
reference Astribot server. The conversion path inside
`async_policy_server.py:QwenPIKMAsyncPolicy._to_robot_29` is:

```
predict_action  → [T, 14]  (delta + normalized dual-arm)
denormalize     → [T, 14]  (inverse of AstribotPipeline)
delta-to-abs    → per-arm xyz + r6d + gripper, anchored on raw state
pack 29-dim     → torso (from state) + left arm + left grip + right arm + right grip
```

## Smoke tests

```bash
# Dataloader: starVLA's standard lerobot loader exercises the Astribot adapter.
python starVLA/dataloader/lerobot_datasets.py \
    --config_yaml examples/Astribot/train_files/starvla_astribot_qwenpikm.yaml

# QwenPIKM forward (KL=0)
python starVLA/model/framework/VLM4A/QwenPIKM.py \
    --config_yaml examples/Astribot/train_files/starvla_astribot_qwenpikm.yaml \
    --kl_loss_weight 0

# QwenPIKM forward with KL on
python starVLA/model/framework/VLM4A/QwenPIKM.py \
    --config_yaml examples/Astribot/train_files/starvla_astribot_qwenpikm.yaml \
    --kl_loss_weight 0.005

# QwenPIKM forward with KI
python starVLA/model/framework/VLM4A/QwenPIKM.py \
    --config_yaml examples/Astribot/train_files/starvla_astribot_qwenpikm.yaml \
    --enable_ki

# Async server end-to-end (subprocess + ws sender + ws receiver)
python examples/Astribot/eval_files/smoke_async_server.py

# Three deeper audits: CLAP encode↔decode roundtrip on real data, pipeline
# forward/inverse exact-equality, and server metadata handshake.
python examples/Astribot/eval_files/smoke_pipeline.py
```
