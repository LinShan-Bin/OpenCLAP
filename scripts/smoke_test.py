"""Smoke tests for the OpenCLAP repo.

Run with::

    PYTHONPATH=. python scripts/smoke_test.py

Verifies (in <60 seconds, on a single GPU):
  1. clap package imports
  2. starVLA framework registry discovers QwenAR / QwenPI / QwenPIKM
  3. DINO_CLAP (stage 1) instantiates and runs one forward+backward pass
  4. starVLA QwenAR class is registered + has the expected API
  5. Sync policy server imports + arg parser builds
  6. All YAML configs (clap-s1, clap-s2, AR, QwenPIKM, LIBERO) load cleanly
"""
from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

# Anchor at the repo root so the test runs cleanly from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch


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
# 1. clap package imports
# ----------------------------------------------------------------------
with section("1) Import clap package"):
    import clap  # noqa: F401
    from clap.modules import (
        ContrastiveDINOLatentActionModel,
        LatentActionModel,
        DualBranchLatentActionModel,
    )
    from clap.model_clap import DINO_CLAP  # noqa: F401
    from clap.dataset_lerobot import LightningLerobot  # noqa: F401

# ----------------------------------------------------------------------
# 2. starVLA framework discovery
# ----------------------------------------------------------------------
with section("2) starVLA framework registry"):
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    from starVLA.model.framework import base_framework as bf
    bf._auto_import_framework_modules()
    keys = sorted(FRAMEWORK_REGISTRY._registry.keys())
    print(f"  registered: {keys}")
    expected = {"QwenAR", "QwenPI", "QwenPIKM"}
    missing = expected - set(keys)
    assert not missing, f"missing frameworks: {missing}"


# ----------------------------------------------------------------------
# 3. DINO_CLAP forward+backward
# ----------------------------------------------------------------------
with section("3) DINO_CLAP instantiation + tiny forward+backward"):
    # Build a minimal stage-1 CLAP model — keeps the heavy DINO backbone
    # but uses a small action-VAE so smoke fits in <30 seconds.
    model = DINO_CLAP(
        image_channels=3,
        action_space="xyz_rpy",
        chunk_size=32,
        clap_model_dim=256,           # paper uses 768 — drop for speed
        clap_latent_dim=128,
        clap_action_vae_dim=256,
        clap_num_latents=128,
        clap_num_t_codes=8,
        clap_patch_size=16,
        clap_enc_blocks=2,
        clap_dec_blocks=2,
        clap_num_heads=4,
        action_layers=3,              # MldVae requires odd
        # The real DINOv3 weights (ViT-B/16). Override at the CLI if
        # they live elsewhere on your machine.
        dino_model_type="dinov3",
        dino_model_variant="dinov3_vitb16",
        dino_model_path="/kpfs-intern/mycroft/models/dinov3",
        dino_weights_path="/kpfs-intern/mycroft/models/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        mse_alpha=1.0,
        vq_beta=1.0,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        max_training_steps=100,
        min_lr_ratio=0.1,
        codebook_lr_scale=1.0,
        task_name="smoke-stage-1",
        stage="stage-1",
        make_data_pair=False,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"  DINO_CLAP params: {n/1e6:.1f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    # Synthetic batch matching the post-AstribotPipeline contract:
    #   - action: [B, T, 14] dual-arm xyz_rpy delta
    #   - state:  [B, 1, 14]
    #   - arm_mask: [B, 2] bool (left arm active, right arm active)
    B, T, D = 2, 32, 14
    batch = {
        "action": torch.randn(B, T, D, device=device) * 0.05,
        "state": torch.randn(B, 1, D, device=device) * 0.05,
        "arm_mask": torch.ones(B, 2, dtype=torch.bool, device=device),
        "robot_id": torch.zeros(B, dtype=torch.long, device=device),
    }

    with torch.no_grad():
        outputs, loss, aux = model.shared_step(batch)
    print(f"  forward loss: {loss.item():.4f}  aux keys: {[k for k, _ in aux]}")
    assert torch.isfinite(loss), "loss is non-finite"

    # Backward pass — verify autograd works through the codebook + decoder.
    loss = model.shared_step(batch)[1]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    print(f"  backward done — {len(grads)} parameters got gradients")
    assert grads, "no parameters received gradients"

# ----------------------------------------------------------------------
# 4. starVLA QwenAR class
# ----------------------------------------------------------------------
with section("4) starVLA QwenAR class is loadable"):
    cls = FRAMEWORK_REGISTRY._registry["QwenAR"]
    print(f"  QwenAR resolved to: {cls.__module__}.{cls.__name__}")
    for method in ("forward", "predict_action"):
        assert hasattr(cls, method), f"QwenAR missing {method}"
    print("  forward + predict_action present")

# ----------------------------------------------------------------------
# 5. Sync policy server imports
# ----------------------------------------------------------------------
with section("5) Sync policy server imports"):
    sys.path.insert(0, ".")
    from examples.Astribot.eval_files import sync_policy_server  # noqa: F401
    from examples.Astribot.eval_files import sync_policy_client  # noqa: F401
    parser = sync_policy_server.build_argparser()
    args = parser.parse_args(["--ckpt_path", "/dev/null"])
    print(f"  parsed sync server args (ckpt_path={args.ckpt_path}, port={args.port})")

# ----------------------------------------------------------------------
# 6. YAML configs parse
# ----------------------------------------------------------------------
with section("6) YAML configs load cleanly"):
    from omegaconf import OmegaConf
    yamls = [
        "clap/configs/clap-s1-l32.yaml",
        "clap/configs/clap-s2-l32.yaml",
        "examples/Astribot/train_files/starvla_astribot_qwenar.yaml",
        "examples/Astribot/train_files/starvla_astribot_qwenpikm.yaml",
        "examples/LIBERO/train_files/starvla_libero_clap_km_l16.yaml",
    ]
    for y in yamls:
        cfg = OmegaConf.load(y)
        # poke at expected keys
        if "framework" in cfg:
            assert "name" in cfg.framework, f"{y}: missing framework.name"
        else:
            assert "model" in cfg, f"{y}: missing model block"
        print(f"  {y} OK")

print("\n[ALL GREEN] smoke tests passed.")

