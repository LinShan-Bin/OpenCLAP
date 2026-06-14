# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenAR Framework
================

Pure autoregressive next-token-prediction port of CLAP-VLA (stage 3).

Mirrors ``clap/model_clap_vla.py`` (the original
Lightning ``CLAP_VLA`` class, ~1090 LOC) line-for-line where it matters:

  - Frozen CLAP action VQ-VAE used as a *solutionizer*: encodes continuous
    actions into ``<ACT_*>`` token strings (model_clap_vla.py L488-563,
    ``encode_action_to_tokens`` / ``encode_visual_to_tokens``).
  - Qwen-VL backbone (Qwen3-VL preferred, Qwen2.5-VL fallback) trained with
    pure cross-entropy over ``[Subtask: ..., Action: <ACT_*>...]``
    (model_clap_vla.py L565-674, ``training_step``).
  - Inference does greedy generation, parses out ``<ACT_*>`` tokens, and
    decodes them through the CLAP codebook + action VAE
    (model_clap_vla.py L732-918, ``predict_action`` / ``_decode_action_tokens``).

Differences from QwenPIKM
-------------------------
  - **No flow-matching action expert** (``framework.use_action_expert =
    false`` per ``configs/clap-s3-l32.yaml``).
  - **No KL term / no reference VLM** (``kl_loss_weight = 0`` per yaml).
  - **No KI gate** (``enable_ki = false``).
  - The VLM is the only trainable submodule.

Action-token vocabulary ranges (must match ``model_clap_vla.py`` L79-81):
  - Qwen2.5-VL: [151665, 153712]
  - Qwen3-VL  : [151936, 153984]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
logging.basicConfig(format="%(message)s", level=logging.INFO)

# HuggingFace / LLaMa-2 IGNORE_INDEX (for masked label positions)
IGNORE_INDEX = -100


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenAR
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenARDefaultConfig:
    """QwenAR framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section. Extra YAML keys are kept as-is.
    """

    # --- Registry identifier ---
    name: str = "QwenAR"

    # === VLM backbone (Qwen3-VL or Qwen2.5-VL with <ACT_*> special tokens) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (must include the <ACT_*> action-token
            # range; see starVLA/model/modules/vlm/tools/add_qwen_special_tokens).
            "base_vlm": "./pretrained/Qwen3-VL-4B-Instruct",
            # "qwen2.5" or "qwen3"; selects which HF class to instantiate.
            "qwen_vl_variant": "qwen3",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa".
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Frozen CLAP action VQ-VAE (used purely as a solutionizer) ===
    tokenizer: dict = field(
        default_factory=lambda: {
            # Path to the CLAP checkpoint (lightning .ckpt). Stage-3 of
            # CLAP-VLA loads the *stage-2* CLAP weights.
            "clap_ckpt": "./clap/ckpts/clap-s2-l32/last.ckpt",
            "clap_chunk_size": 32,
            "num_t_codes": 8,         # action codes per arm
            "visual_t_codes": 8,
            "num_latents": 512,       # codebook size
            "model_dim": 768,
            "latent_dim": 128,
            "action_vae_dim": 512,
            "patch_size": 16,
            "enc_blocks": 12,
            "dec_blocks": 12,
            "num_heads": 12,
            "dropout": 0.0,
            "image_channels": 3,
            "action_dim_per_arm": 7,
            # DINO is required at __init__ time but ``action_vq_encode`` and
            # the codebook lookup never touch it; we stub it out below so no
            # DINO weights are needed.
            "dino_model_type": "dinov2",
            "dino_model_variant": "vits14",
            "dino_model_path": "facebook/dinov2-small",
            "dino_weights_path": None,
        }
    )

    # === Training-time flags (mirror ``configs/clap-s3-l32.yaml``) ===
    flags: dict = field(
        default_factory=lambda: {
            # Re-init the freshly-added <ACT_*> token embeddings with mean +
            # 2% std noise of the original vocab. Mirrors model_clap_vla.py
            # L267-341 (``_initialize_new_token_embeddings``).
            "init_new_token_embeddings": True,
            # Append wrist images (right then left) after the head image,
            # giving the VLM up to 3 views per sample.
            "use_wrist_camera": True,
            # Per-step Bernoulli drop probability for the wrist views, as in
            # the original co-training recipe.
            "drop_wrist_prob": 0.5,
            # Per-step probability of zeroing out the state input to the VLM.
            # AR has no state stream, so this is informational only here.
            "drop_state_prob": 0.5,
            # Pure-AR has no flow-matching expert and no LoRA.
            "enable_lora": False,
            "use_action_expert": False,
            # Single-arm pipeline (LIBERO / DROID): enables
            # ``LiberoDroidStyleTokenPipeline`` to pad single-arm 7-D actions
            # into the dual-arm 14-D layout CLAP expects. Off by default.
            "single_arm": False,
            "single_arm_layout": "single_right",
            # Optional max generation length at inference (matches L784).
            "max_new_tokens": 512,
        }
    )


# ──────────────────────────────────────────────────────────────────────
#  Framework
# ──────────────────────────────────────────────────────────────────────
@FRAMEWORK_REGISTRY.register("QwenAR")
class Qwen_AR(baseframework):
    """Pure autoregressive CLAP-VLA stage-3 in starVLA framework form.

    Parameters & dataflow follow ``CLAP_VLA`` in
    ``clap/model_clap_vla.py`` exactly.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenARDefaultConfig, config)

        # --- VLM backbone ---------------------------------------------------
        # Reuse starVLA's VLM dispatcher for consistency with other Qwen*
        # frameworks. The dispatcher loads Qwen3-VL or Qwen2.5-VL based on
        # ``base_vlm`` and exposes a unified ``.model`` / ``.processor`` API.
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # Left-padding is required for batched generation (model_clap_vla.py L229).
        self.qwen_vl_interface.processor.tokenizer.padding_side = "left"

        qcfg = self.config.framework.qwenvl
        self.qwen_vl_variant = str(qcfg.get("qwen_vl_variant", "qwen3"))

        # --- Tokenizer (frozen CLAP) ---------------------------------------
        tk = self.config.framework.tokenizer
        self.clap_chunk_size = int(tk.get("clap_chunk_size", 32))
        self.num_t_codes = int(tk.get("num_t_codes", 8))
        self.num_action_codes = self.num_t_codes
        self.clap_num_latents = int(tk.get("num_latents", 512))
        self.clap_latent_dim = int(tk.get("latent_dim", 128))
        self.action_dim_per_arm = int(tk.get("action_dim_per_arm", 7))
        self.max_action_dim = self.action_dim_per_arm
        self._init_clap_tokenizer(tk)

        # --- Action token bookkeeping --------------------------------------
        # The tokenizer of every ``-Action`` checkpoint already contains the
        # ``<ACT_*>`` range. We resolve their ids once for fast masking.
        self.action_tokens: List[str] = [f"<ACT_{i}>" for i in range(self.clap_num_latents)]
        self._resolve_action_token_ids()

        # --- Flags ----------------------------------------------------------
        flags = self.config.framework.flags
        self.use_wrist_camera = bool(flags.get("use_wrist_camera", True))
        self.drop_wrist_prob = float(flags.get("drop_wrist_prob", 0.5))
        self.drop_state_prob = float(flags.get("drop_state_prob", 0.5))
        self.init_new_token_embeddings = bool(flags.get("init_new_token_embeddings", True))
        self.single_arm = bool(flags.get("single_arm", False))
        self.single_arm_layout = str(flags.get("single_arm_layout", "single_right"))
        self.max_new_tokens = int(flags.get("max_new_tokens", 512))

        # Optional Astribot / single-arm transforms; loaded lazily so a stripped
        # repo without ``clap.data_transform`` still imports cleanly.
        self._pipeline = None  # lazy

        # ``<|im_start|>`` and ``assistant\n`` ids needed for label masking
        # — same logic as model_clap_vla.py L252-254.
        tok = self.qwen_vl_interface.processor.tokenizer
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.assistant_len = len(tok.encode("assistant\n", add_special_tokens=False))

        # Track whether we have re-init'd the new token embeddings yet (only
        # done once at the start of the first training forward).
        self._inited_new_token_embeds = False

    # ──────────────────────────────────────────────────────────────────
    # CLAP solutionizer init (frozen, kept outside nn.Module submodule tree)
    # ──────────────────────────────────────────────────────────────────
    def _init_clap_tokenizer(self, tk_cfg) -> None:
        """Instantiate a frozen CLAP encoder/decoder used purely for action
        discretisation + decoding. DINO is stubbed out — ``action_vq_encode``
        and codebook lookups never touch it (mirrors QwenPIKM)."""
        try:
            from clap import modules as _clap_modules
            from clap.modules import ContrastiveDINOLatentActionModel  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "QwenAR requires `clap.modules.ContrastiveDINOLatentActionModel`. "
                "OpenCLAP/clap should be on PYTHONPATH (it is a port of "
                "latent_action_model/genie)."
            ) from e

        original_loader = _clap_modules.clap.load_dino_encoder
        _clap_modules.clap.load_dino_encoder = lambda **_: nn.Identity()
        try:
            clap_module = ContrastiveDINOLatentActionModel(
                in_dim=int(tk_cfg.get("image_channels", 3)),
                model_dim=int(tk_cfg.get("model_dim", 768)),
                chunk_size=self.clap_chunk_size,
                latent_dim=int(tk_cfg.get("latent_dim", 128)),
                action_vae_dim=int(tk_cfg.get("action_vae_dim", 512)),
                max_action_dim=self.action_dim_per_arm,
                num_latents=self.clap_num_latents,
                num_t_codes=self.num_t_codes,
                visual_t_codes=int(tk_cfg.get("visual_t_codes", 8)),
                patch_size=int(tk_cfg.get("patch_size", 16)),
                enc_blocks=int(tk_cfg.get("enc_blocks", 12)),
                dec_blocks=int(tk_cfg.get("dec_blocks", 12)),
                num_heads=int(tk_cfg.get("num_heads", 12)),
                dropout=float(tk_cfg.get("dropout", 0.0)),
                dino_model_type=str(tk_cfg.get("dino_model_type", "dinov2")),
                dino_model_variant=str(tk_cfg.get("dino_model_variant", "vits14")),
                dino_model_path=str(tk_cfg.get("dino_model_path", "facebook/dinov2-small")),
                dino_weights_path=tk_cfg.get("dino_weights_path", None),
            ).float()
        finally:
            _clap_modules.clap.load_dino_encoder = original_loader

        ckpt_path = tk_cfg.get("clap_ckpt", None)
        if ckpt_path is None:
            raise ValueError("framework.tokenizer.clap_ckpt must be set for QwenAR")
        logger.info(f"[QwenAR] Loading frozen CLAP weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("clap.", ""): v for k, v in sd.items()}
        sd = {
            k: v
            for k, v in sd.items()
            if "astribot_pipeline" not in k
            and "action_denormalization" not in k
            and not k.startswith("dino_encoder")
        }
        missing, unexpected = clap_module.load_state_dict(sd, strict=False)
        non_dino_missing = [k for k in missing if not k.startswith("dino_encoder")]
        logger.info(
            f"[QwenAR] CLAP load_state_dict: missing={len(non_dino_missing)} "
            f"(plus DINO stubs) unexpected={len(unexpected)}"
        )
        clap_module.requires_grad_(False)
        clap_module.eval()
        # Keep CLAP outside the nn.Module submodule tree so DeepSpeed's optimizer
        # doesn't see frozen CLAP params (same trick as QwenPIKM).
        object.__setattr__(self, "clap", clap_module)

    # ──────────────────────────────────────────────────────────────────
    def _resolve_action_token_ids(self) -> None:
        """Cache (min, max) token-id range for the ``<ACT_*>`` vocabulary.

        Per ``model_clap_vla.py`` the canonical ranges are:
          - Qwen2.5-VL: [151665, 153712]
          - Qwen3-VL  : [151936, 153984]
        We still resolve via the tokenizer (for safety) but log a warning if
        the resolved range disagrees with the canonical constants.
        """
        tok = self.qwen_vl_interface.processor.tokenizer
        ids = [tok.convert_tokens_to_ids(t) for t in self.action_tokens]
        if any(i is None or i == tok.unk_token_id for i in ids):
            # Fall back to canonical constants if the tokenizer hasn't been
            # extended (e.g. plain Qwen3-VL without <ACT_*> tokens). The trainer
            # is expected to add them in this case.
            if self.qwen_vl_variant == "qwen3":
                base = 151936
            else:
                base = 151665
            ids = [base + i for i in range(self.clap_num_latents)]
            logger.warning(
                f"[QwenAR] tokenizer did not contain <ACT_*> tokens; falling back to "
                f"canonical range starting at {base}. The base VLM checkpoint must "
                f"have these special tokens added before training."
            )
        self.action_token_ids: List[int] = list(ids)
        self.action_token_min = min(ids)
        self.action_token_max = max(ids)
        # Cross-check against the constants in ``model_clap_vla.py``.
        canonical = (151665, 153712) if self.qwen_vl_variant == "qwen2.5" else (151936, 153984)
        # Stage-3 numbers a 512-codebook → 512 tokens. Range == [base, base + 511]
        # ⊂ canonical reservation; we only assert it sits *inside* the canonical block.
        if not (self.action_token_min >= canonical[0] and self.action_token_max <= canonical[1]):
            logger.warning(
                f"[QwenAR] resolved <ACT_*> range "
                f"[{self.action_token_min}, {self.action_token_max}] is outside the "
                f"canonical {self.qwen_vl_variant} block {canonical}. Double-check the "
                f"VLM checkpoint."
            )

    # ──────────────────────────────────────────────────────────────────
    # Optional dataset-side pipeline (LIBERO / Astribot single arm)
    # ──────────────────────────────────────────────────────────────────
    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self.single_arm:
            try:
                from clap.data_transform_single_arm import LiberoDroidStyleTokenPipeline
            except ImportError as e:
                raise ImportError(
                    "single_arm=True requires clap.data_transform_single_arm.LiberoDroidStyleTokenPipeline."
                ) from e
            self._pipeline = LiberoDroidStyleTokenPipeline(
                arm_layout=self.single_arm_layout,
            )
        else:
            try:
                from clap.data_transform import AstribotPipeline
            except ImportError as e:
                raise ImportError(
                    "QwenAR's default Astribot batches require clap.data_transform.AstribotPipeline."
                ) from e
            self._pipeline = AstribotPipeline()
        return self._pipeline

    # ──────────────────────────────────────────────────────────────────
    # Mean+noise re-initialisation of new <ACT_*> embeddings
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _initialize_new_token_embeddings(self) -> None:
        """Mean + 2% std noise re-init of <ACT_*> embeddings.
        Direct port of model_clap_vla.py L267-341."""
        embed = self.qwen_vl_interface.model.model.language_model.embed_tokens
        ids = self.action_token_ids
        original_vocab_size = len(self.qwen_vl_interface.processor.tokenizer)
        num_tokens_for_mean = min(original_vocab_size // 2, 50000)
        existing = embed.weight[:num_tokens_for_mean]
        mean = existing.mean(dim=0, keepdim=True)
        std = existing.std(dim=0, keepdim=True)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(mean, src=0)
            torch.distributed.broadcast(std, src=0)
        noise = torch.randn(
            len(ids), embed.weight.size(-1),
            device=embed.weight.device, dtype=embed.weight.dtype,
        ) * std * 0.02
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(noise, src=0)
        for i, tid in enumerate(ids):
            embed.weight[tid] = mean.squeeze(0) + noise[i]
        logger.info(
            f"[QwenAR] re-initialised {len(ids)} <ACT_*> embeddings "
            f"(mean norm {mean.norm().item():.4f}, std norm {std.norm().item():.4f})"
        )

    # ──────────────────────────────────────────────────────────────────
    # Action -> <ACT_*> string  (training)
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_action_to_tokens(self, action: torch.Tensor) -> List[str]:
        """Continuous actions [B, T, D] → list of ``<ACT_*>`` strings.

        Mirrors model_clap_vla.py L488-529: encodes left/right halves of the
        14-dim dual-arm action separately and concatenates as
        ``[right_indices, left_indices]`` (right first, the legacy stage-3
        wire format)."""
        target_device = next(self.qwen_vl_interface.model.parameters()).device
        if next(self.clap.parameters()).device != target_device:
            self.clap.to(target_device)
        action = action.to(target_device).float()
        B, T, D = action.shape
        d_per_arm = self.action_dim_per_arm

        # Resample window to CLAP's training chunk size (32).
        if T == self.clap_chunk_size:
            action_clap = action
        elif T > self.clap_chunk_size:
            action_clap = action[:, -self.clap_chunk_size:, :]
        else:
            pad = self.clap_chunk_size - T
            action_clap = torch.cat([action, action[:, -1:, :].expand(-1, pad, -1)], dim=1)

        if D >= 2 * d_per_arm:
            left = action_clap[..., :d_per_arm]
            right = action_clap[..., d_per_arm:2 * d_per_arm]
            action_concat = torch.cat([left, right], dim=0)  # [2B, T, d_per_arm]
            with torch.autocast("cuda", enabled=False):
                indices = self.clap.action_vq_encode(action_concat)["indices"]
            indices = indices.reshape(2, B, self.num_action_codes)
            left_idx, right_idx = indices[0].cpu().numpy(), indices[1].cpu().numpy()
            return [
                "".join(f"<ACT_{int(i)}>" for i in right_idx[b])
                + "".join(f"<ACT_{int(i)}>" for i in left_idx[b])
                for b in range(B)
            ]

        # Single-arm fallback: only one arm encoded, no concatenation.
        with torch.autocast("cuda", enabled=False):
            indices = self.clap.action_vq_encode(action_clap[..., :d_per_arm])["indices"]
        idx_np = indices.cpu().numpy()
        return ["".join(f"<ACT_{int(i)}>" for i in idx_np[b]) for b in range(B)]

    @torch.no_grad()
    def encode_visual_to_tokens(self, videos: torch.Tensor) -> List[str]:
        """Two-frame video → ``<ACT_*>`` strings via CLAP's visual VQ.
        Mirrors model_clap_vla.py L531-563 (used for human data)."""
        target_device = next(self.qwen_vl_interface.model.parameters()).device
        if next(self.clap.parameters()).device != target_device:
            self.clap.to(target_device)
        videos = videos.to(target_device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.clap.visual_vq_encode(videos)
        indices = outputs["indices"]  # [B, num_i_codes + num_t_codes]
        num_i = self.clap.num_i_codes
        num_t = self.clap.num_t_codes
        action = indices[:, num_i:].cpu().numpy()  # [B, 2 * num_t_codes]
        left = action[:, :num_t]
        right = action[:, num_t:]
        action = np.concatenate([right, left], axis=1)
        return [
            "".join(f"<ACT_{int(i)}>" for i in action[b])
            for b in range(action.shape[0])
        ]

    # ──────────────────────────────────────────────────────────────────
    # Prompt builder (matches model_clap_vla.py L350-486 exactly)
    # ──────────────────────────────────────────────────────────────────
    def build_qwenvl_inputs(
        self,
        images: torch.Tensor,
        instructions: List[str],
        solutions: Optional[List[str]] = None,
        robot_types: Optional[List[str]] = None,
        robot_ids: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build VLM inputs from a tensor of multi-view images.

        Output prompt format (verbatim from model_clap_vla.py):
            system : "You are a helpful assistant."
            user   : <images> + "Robot category: <type>\\n
                     Control the robot to do the task: <instruction>\\n
                     Please output the subtask and the action tokens."
            assist : "Subtask: <subtask>\\nAction: <ACT_*>..."     (only at training)
        """
        from clap.robot_prompt import expand_robot_types, robot_prompt_line

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None  # type: ignore

        B, N, _, _, _ = images.shape
        assert len(instructions) == B, (
            f"images B={B} != instructions length {len(instructions)}"
        )
        robot_type_names = expand_robot_types(robot_types, robot_ids, B)

        tensor_images = [
            [images[b, n] for n in range(N)] for b in range(B)
        ]

        messages = []
        for imgs, instruction, robot_type in zip(tensor_images, instructions, robot_type_names):
            content = [{"type": "image", "image": img * 255.0} for img in imgs]
            content.append({
                "type": "text",
                "text": (
                    f"{robot_prompt_line(robot_type)}\n"
                    f"Control the robot to do the task: {instruction}\n"
                    "Please output the subtask and the action tokens."
                ),
            })
            msg = [
                {"role": "system",
                 "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": content},
            ]
            if solutions is not None:
                msg.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": solutions[len(messages)]}],
                })
            messages.append(msg)

        processor = self.qwen_vl_interface.processor

        if self.qwen_vl_variant == "qwen2.5":
            if process_vision_info is None:
                raise ImportError("qwen_vl_utils.process_vision_info is required for qwen2.5")
            texts = [
                processor.apply_chat_template(
                    msg, tokenize=False,
                    add_generation_prompt=(solutions is None),
                    do_convert_rgb=False,
                )
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            batch_inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                do_convert_rgb=False,
            )
        elif self.qwen_vl_variant == "qwen3":
            batch_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=(solutions is None),
                return_dict=True,
                return_tensors="pt",
                do_convert_rgb=False,
            )
        else:
            raise ValueError(f"Unknown qwen_vl_variant: {self.qwen_vl_variant!r}")

        target_device = self.qwen_vl_interface.model.device
        batch_inputs = {
            k: (v.to(target_device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch_inputs.items()
        }

        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            for b in range(B):
                seq = batch_inputs["input_ids"][b]
                im_start_pos = (seq == self.im_start_id).nonzero(as_tuple=True)[0]
                if len(im_start_pos) > 0:
                    last = im_start_pos[-1].item()
                    mask_until = last + 1 + self.assistant_len
                    labels[b, :mask_until] = IGNORE_INDEX
                else:
                    labels[b, :] = IGNORE_INDEX
            labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs

    # ──────────────────────────────────────────────────────────────────
    # Batch shaping helpers
    # ──────────────────────────────────────────────────────────────────
    def _stack_images(self, examples: List[dict], device) -> torch.Tensor:
        """Pack ``example["image"]`` (List[PIL] or np.ndarray) into [B, N, C, H, W]
        in the order [head, right_wrist, left_wrist] with optional dropout
        per-sample (model_clap_vla.py L606-624).
        """
        # Each example carries `image: List[PIL.Image]` (model-agnostic data
        # contract in starVLA dataloaders). For Astribot the ordering is
        # [head, right, left]; LIBERO/DROID are typically [head] only.
        from PIL import Image

        def _pil_to_tensor(img):
            if isinstance(img, torch.Tensor):
                t = img
                if t.ndim == 3 and t.shape[0] not in (1, 3):
                    t = t.permute(2, 0, 1)
            elif isinstance(img, np.ndarray):
                t = torch.from_numpy(img)
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
            elif isinstance(img, Image.Image):
                t = torch.from_numpy(np.asarray(img))
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            return t.float() / 255.0 if t.dtype == torch.uint8 else t.float()

        max_views = 1 + (2 if self.use_wrist_camera else 0)
        per_sample = []
        for ex in examples:
            imgs = ex["image"]
            tensors = [_pil_to_tensor(im) for im in imgs[:max_views]]
            # Drop wrists with prob `drop_wrist_prob` at training time.
            if self.training and self.use_wrist_camera and self.drop_wrist_prob > 0:
                if len(tensors) > 1 and torch.rand(()).item() < self.drop_wrist_prob:
                    tensors = tensors[:1]
            # Pad to max_views by repeating the head image so batches stay rectangular.
            while len(tensors) < max_views:
                tensors.append(tensors[0].clone())
            per_sample.append(torch.stack(tensors, dim=0))
        return torch.stack(per_sample, dim=0).to(device)

    @staticmethod
    def _extract_robot_types(examples: List[dict]) -> Optional[List[str]]:
        if not examples or "robot_type" not in examples[0]:
            return None
        return [str(ex.get("robot_type", "franka")) for ex in examples]

    # ──────────────────────────────────────────────────────────────────
    # forward = pure NTP loss
    # ──────────────────────────────────────────────────────────────────
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward: cross-entropy over [Subtask, <ACT_*>...]."""
        if not isinstance(examples, list):
            examples = [examples]
        if self.init_new_token_embeddings and not self._inited_new_token_embeds:
            try:
                self._initialize_new_token_embeddings()
            except Exception as e:  # never block training on a re-init failure
                logger.warning(f"[QwenAR] new-token re-init failed: {e}")
            self._inited_new_token_embeds = True

        device = self.qwen_vl_interface.model.device

        # Pull instructions / actions / state / robot ids from raw examples.
        instructions = [ex["lang"] for ex in examples]
        subtasks = [ex.get("subtask", "") for ex in examples]
        robot_types = self._extract_robot_types(examples)
        robot_ids = [ex.get("robot_id", 0) for ex in examples] if "robot_id" in examples[0] else None

        actions = torch.tensor(
            np.array([ex["action"] for ex in examples]),
            device=device, dtype=torch.float32,
        )

        # CLAP-discretise (robot data only). Human data with no actions falls
        # through to ``encode_visual_to_tokens`` below.
        is_human = torch.tensor(
            [int(rid) == 2 for rid in (robot_ids or [0] * len(examples))],
            device=device, dtype=torch.bool,
        )
        is_robot = ~is_human
        action_tokens_robot, action_tokens_human = [], []
        if is_robot.any():
            action_tokens_robot = self.encode_action_to_tokens(actions[is_robot])
        if is_human.any():
            # Two-frame head video for visual codes; expects [B, 2, C, H, W]
            head_videos = torch.stack([
                torch.from_numpy(np.array(ex["head_video"])) for ex, h in zip(examples, is_human) if h
            ]).to(device)
            action_tokens_human = self.encode_visual_to_tokens(head_videos)

        solutions = []
        h_cum = is_human.cumsum(dim=0) - 1
        r_cum = is_robot.cumsum(dim=0) - 1
        for i, sub in enumerate(subtasks):
            tok = (action_tokens_human[h_cum[i]] if is_human[i]
                   else action_tokens_robot[r_cum[i]])
            solutions.append(f"Subtask: {sub}\nAction: {tok}")

        images = self._stack_images(examples, device)
        qwen_inputs = self.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            solutions=solutions,
            robot_types=robot_types,
            robot_ids=robot_ids,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        loss = outputs.loss
        if loss is None or torch.isnan(loss):
            loss = torch.tensor(0.0, device=device)
        return {"action_loss": loss}

    # ──────────────────────────────────────────────────────────────────
    # predict_action  (greedy)
    # ──────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> Dict:
        """Greedy decode → parse <ACT_*> ids → CLAP codebook decode.

        Returns:
            dict with the contract of model_clap_vla.py L755-808:
              - subtasks       : List[str]
              - actions        : torch.Tensor [B, T, D]
              - action_tokens  : List[List[int]]   (CLAP indices)
              - generated_texts: List[str]
            Plus ``normalized_actions`` (alias for ``actions``) so existing
            starVLA eval code that pulls that key continues to work.
        """
        if not isinstance(examples, list):
            examples = [examples]
        instructions = [ex["lang"] for ex in examples]
        robot_types = self._extract_robot_types(examples)
        robot_ids = [ex.get("robot_id", 0) for ex in examples] if "robot_id" in examples[0] else None

        device = self.qwen_vl_interface.model.device
        images = self._stack_images(examples, device)
        qwen_inputs = self.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            solutions=None,
            robot_types=robot_types,
            robot_ids=robot_ids,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_texts = self.qwen_vl_interface.processor.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        batch_clap_indices = self._extract_action_token_ids(generated_ids)
        actions = self._decode_action_tokens(batch_clap_indices)
        subtasks = self._extract_subtasks(generated_texts)

        actions_tensor = torch.from_numpy(actions)
        return {
            "subtasks": subtasks,
            "actions": actions_tensor,
            "action_tokens": batch_clap_indices,
            "generated_texts": generated_texts,
            # starVLA serving / eval looks for this key.
            "normalized_actions": actions,
        }

    # ──────────────────────────────────────────────────────────────────
    # Action-token parsing + decoding  (model_clap_vla.py L811-918)
    # ──────────────────────────────────────────────────────────────────
    def _extract_action_token_ids(self, generated_ids: torch.LongTensor) -> List[List[int]]:
        mask = (generated_ids >= self.action_token_min) & (generated_ids <= self.action_token_max)
        results: List[List[int]] = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            vlm_tokens = generated_ids[b, idx].tolist()
            results.append([t - self.action_token_min for t in vlm_tokens])
        return results

    def _decode_action_tokens(self, batch_clap_indices: List[List[int]]) -> np.ndarray:
        """Decode [right_indices, left_indices] CLAP indices to [B, T, D=2*d]
        continuous actions through ``clap.action_vae``."""
        d = self.action_dim_per_arm
        device = next(self.clap.parameters()).device
        batch_actions: List[np.ndarray] = []
        for clap_indices in batch_clap_indices:
            if len(clap_indices) == 0:
                batch_actions.append(np.zeros((1, d * 2), dtype=np.float32))
                continue
            indices_tensor = torch.tensor(clap_indices, device=device, dtype=torch.long)
            n = self.num_action_codes
            if indices_tensor.numel() >= 2 * n:
                right = indices_tensor[:n]
                left = indices_tensor[n:2 * n]
            else:
                right = indices_tensor[:n]
                left = right
            left_lat = self.clap.vq_t.codebook(left).unsqueeze(0)
            right_lat = self.clap.vq_t.codebook(right).unsqueeze(0)
            combined = torch.cat([left_lat, right_lat], dim=0)  # [2, n, latent_dim]
            z_q = combined.permute(1, 0, 2)
            T = 1
            action_recon = self.clap.action_vae.decode(z_q, [T, T])  # [2, T, d]
            left_a = action_recon[0].detach().cpu().float().numpy()
            right_a = action_recon[1].detach().cpu().float().numpy()
            batch_actions.append(np.concatenate([left_a, right_a], axis=-1))
        max_len = max(a.shape[0] for a in batch_actions)
        out = []
        for a in batch_actions:
            if a.shape[0] < max_len:
                a = np.concatenate(
                    [a, np.zeros((max_len - a.shape[0], d * 2), dtype=a.dtype)], axis=0
                )
            out.append(a)
        return np.stack(out, axis=0)

    @staticmethod
    def _extract_subtasks(generated_texts: List[str]) -> List[str]:
        subtasks = []
        for text in generated_texts:
            if "Subtask:" in text:
                start = text.find("Subtask:") + len("Subtask:")
                a = text.find("Action:", start)
                t = text.find("<ACT_", start)
                if a != -1:
                    sub = text[start:a].strip()
                elif t != -1:
                    sub = text[start:t].strip()
                else:
                    sub = text[start:].strip()
            else:
                t = text.find("<ACT_")
                sub = text[:t].strip() if t != -1 else text.strip()
            subtasks.append(sub)
        return subtasks


# ──────────────────────────────────────────────────────────────────────
# Standalone sanity-check  (no GPU forward — just config + instantiation):
#   python OpenCLAP/starVLA/model/framework/VLM4A/QwenAR.py \
#       --config_yaml OpenCLAP/examples/Astribot/train_files/starvla_astribot_qwenar.yaml
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    model = Qwen_AR(cfg)
    print("[QwenAR smoke] action token range:", model.action_token_min, model.action_token_max)
    print("[QwenAR smoke] num_action_codes :", model.num_action_codes)
    print("[QwenAR smoke] OK")
