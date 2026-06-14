# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenPIKM Framework
==================

QwenPI + Knowledge Matching (KL).

Same architecture as ``QwenPI`` (layer-wise cross-DiT flow-matching action
head on top of Qwen3-VL hidden states), with one extra training-time term:

    total_loss = action_loss + kl_loss_weight * KL(student, teacher)

The KL is computed *on GT action-token positions* of an auxiliary forward where
the assistant turn carries the GT continuous actions discretised into
``<ACT_*>`` tokens by a frozen **CLAP action VQ-VAE** — same prompt convention
as ``clap/model_clap_vla_fm.py``.

Two variants of KL are supported:
  - ``kl_type: "kl"``         — forward KL ``KL(student || teacher)`` (mode-covering)
  - ``kl_type: "reverse_kl"`` — reverse KL ``KL(teacher || student)`` (mode-seeking),
                                 the variant CLAP-VLA-FM uses.

Requirements
------------
- ``framework.qwenvl.base_vlm`` must be a Qwen3-VL checkpoint whose tokenizer
  contains the ``<ACT_*>`` action-token range (e.g. the same VLM used to train
  CLAP-VLA-FM, such as ``latent_action_model/ckpts/clap-s3-l32/qwen_model-stepXXXX``).
- ``framework.knowledge_matching.clap.clap_ckpt`` must point at the
  matching CLAP ``.ckpt`` (the encoder side is frozen).
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.share_tools import (
    merge_framework_config,
    populate_layerwise_dit_cfg,
    resolve_vl_layer_selection,
)
from starVLA.model.framework.VLM4A.QwenPI import IGNORE_INDEX, QwenPIDefaultConfig, Qwen_PI
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _NullCtx:
    """No-op context manager (avoids ``contextlib.nullcontext`` dependency)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class QwenPIKMDefaultConfig(QwenPIDefaultConfig):
    """QwenPIKM = QwenPI + knowledge-matching KL on GT-action-token positions."""

    name: str = "QwenPIKM"

    # === Knowledge Insulating (KI) ===
    # When True, the VLM forward used by the flow-matching head runs under
    # ``torch.no_grad()`` so no gradients flow back into the VLM. Only the
    # action expert is trained. Mutually exclusive with KL (kl_loss_weight > 0)
    # — same constraint as ``CLAP_VLA_FM`` in latent_action_model. Default
    # False to preserve the existing QwenPIKM training recipe.
    enable_ki: bool = False

    # === Knowledge-matching KL constraint ===
    knowledge_matching: dict = field(
        default_factory=lambda: {
            # Multiplier on the KL term added to the total loss. Set to 0 to
            # disable the regulariser entirely (no reference VLM loaded, no
            # CLAP, no overhead). Default 0.005.
            "kl_loss_weight": 0.005,

            # "kl"         : KL(student || teacher), forward, mode-covering.
            # "reverse_kl" : KL(teacher || student), reverse, mode-seeking.
            #                CLAP-VLA-FM uses this variant.
            "kl_type": "reverse_kl",

            # Arm layout the action vector encodes:
            #   "single_right" — action is [T, D_per_arm]; encode the single arm.
            #   "dual"         — action is [T, 2 * D_per_arm]; split into left/right
            #                    and encode both arms in one CLAP forward (mirrors
            #                    ``encode_action_to_tokens`` in CLAP-VLA-FM with
            #                    default arm_mask = [1, 1]).
            "arm_layout": "single_right",

            # CLAP action-VQ encoder used to turn GT continuous actions into
            # `<ACT_*>` token strings. Defaults match the clap-s3-l32 config
            # in clap/configs/clap-s3-l32.yaml; override
            # per-checkpoint as needed.
            "clap": {
                "clap_ckpt": "./pretrained/clap-s3-l32/clap.ckpt",
                "model_dim": 768,
                "latent_dim": 128,
                "action_vae_dim": 512,
                # codebook size (number of distinct VQ latents). For
                # clap-s3-l32 this is 512.
                "num_latents": 512,
                # number of action-token slots per sample (== "<ACT_*>" tokens
                # emitted into the prompt). For clap-s3-l32 this is 8.
                "num_t_codes": 8,
                "visual_t_codes": 8,
                "patch_size": 16,
                "enc_blocks": 12,
                "dec_blocks": 12,
                "num_heads": 12,
                "dropout": 0.0,
                # action_dim_per_arm; CLAP single-arm uses the full action_dim.
                "action_dim_per_arm": None,  # falls back to action_model.action_dim
                # Optional per-arm column indices into the model's native action
                # vector when ``arm_layout == "dual"`` and the dataset's column
                # order is *not* the contiguous ``[left | right]`` layout CLAP
                # expects. Provide a {"left": [...], "right": [...]} dict of
                # column indices (each list of length ``action_dim_per_arm``)
                # and ``_actions_to_solutions`` will gather/reorder columns
                # before handing the per-arm slices to CLAP.
                #
                # Used by RoboTwin (Agilex), whose 14-D action is laid out as
                # [L_joints(6) | R_joints(6) | L_grip(1) | R_grip(1)] but CLAP
                # was trained on per-arm [joints(6) | grip(1)].
                #
                # Affects ONLY the KL discretiser path; the flow-matching head
                # and ``predict_action`` continue to see the dataset's native
                # column order, so eval/serving code is unchanged.
                "dual_arm_indices": None,
                # CLAP was trained on a fixed chunk length (32 for libero_franka).
                # The flow-matching head can use a different action_horizon
                # (e.g. 16); we pad/truncate the action window to
                # `clap_chunk_size` only for the KL solutionizer path.
                "clap_chunk_size": 32,
                "image_channels": 3,
                # DINO encoder is needed at __init__ time but `action_vq_encode`
                # never touches it; QwenPIKM stubs it out so no DINO weights
                # are required.
                "dino_model_type": "dinov2",
                "dino_model_variant": "vits14",
                "dino_model_path": "facebook/dinov2-small",
                "dino_weights_path": None,
            },

            # The format string used to wrap the action-token sequence into the
            # assistant turn. Must end with the token list and avoid extra
            # tokens that would dilute the KL signal.
            "solution_template": "Action: {action_tokens}",
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPIKM")
class Qwen_PI_KM(Qwen_PI):
    """QwenPI + KL knowledge matching against a frozen reference VLM,
    using a frozen CLAP action VQ-VAE as the action solutionizer."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # We replicate QwenPI's setup (rather than call super().__init__) so we
        # can swap the merged-config dataclass to QwenPIKMDefaultConfig.
        from starVLA.model.framework.base_framework import baseframework
        baseframework.__init__(self)

        self.config = merge_framework_config(QwenPIKMDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Resolve DiT depth and which VLM layers feed it (defaults preserve
        # the original behaviour: one DiT layer per VLM layer, in order).
        num_dit_layers, vl_layer_indices = resolve_vl_layer_selection(
            self.config, num_vl_layers=num_vl_layers
        )
        self.vl_layer_indices: List[int] = vl_layer_indices

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=llm_hidden_size,
            num_dit_layers=num_dit_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # ── KI (Knowledge Insulating) ─────────────────────────────────
        self.enable_ki = bool(getattr(self.config.framework, "enable_ki", False))

        # ── Knowledge-matching KL setup ───────────────────────────────
        km_cfg = getattr(self.config.framework, "knowledge_matching", None)
        self.kl_loss_weight = float(km_cfg.kl_loss_weight) if km_cfg is not None else 0.0
        self.kl_type = str(km_cfg.kl_type) if km_cfg is not None else "reverse_kl"
        self.solution_template = (
            str(km_cfg.solution_template) if km_cfg is not None else "Action: {action_tokens}"
        )
        self.km_arm_layout = (
            str(getattr(km_cfg, "arm_layout", "single_right")) if km_cfg is not None else "single_right"
        )
        if self.kl_type not in ("kl", "reverse_kl"):
            raise ValueError(
                f"framework.knowledge_matching.kl_type must be 'kl' or 'reverse_kl', "
                f"got {self.kl_type!r}"
            )
        if self.km_arm_layout not in ("single_right", "dual"):
            raise ValueError(
                f"framework.knowledge_matching.arm_layout must be 'single_right' or 'dual', "
                f"got {self.km_arm_layout!r}"
            )
        # Optional per-arm column indices for ``arm_layout == "dual"``. See the
        # field docstring on ``QwenPIKMDefaultConfig.knowledge_matching`` for
        # when this is needed. Validation against ``action_dim_per_arm`` happens
        # after CLAP is instantiated (we need ``self.clap.max_action_dim``).
        self.km_dual_arm_indices = (
            getattr(km_cfg.clap, "dual_arm_indices", None) if km_cfg is not None else None
        )
        if self.enable_ki and self.kl_loss_weight > 0:
            # Same constraint as clap/model_clap_vla_fm.py:
            # KI freezes the VLM, KL regularises it — they cannot coexist.
            raise ValueError(
                "QwenPIKM: enable_ki=True is mutually exclusive with kl_loss_weight>0. "
                "Set knowledge_matching.kl_loss_weight=0 when enabling KI."
            )

        if self.kl_loss_weight > 0:
            logger.info(
                f"[QwenPIKM] KL enabled: kl_type={self.kl_type}, "
                f"kl_loss_weight={self.kl_loss_weight}"
            )
            # Build the frozen reference VLM and CLAP solutionizer *outside*
            # this module's submodule tree (via object.__setattr__) so their
            # parameters never appear in `model.parameters()`. Otherwise
            # DeepSpeed's ZeRO optimizer creates an all-frozen param group and
            # crashes flattening with "expected a non-empty list of Tensors".
            self._init_clap_solutionizer(km_cfg.clap)
            ref_vlm = copy.deepcopy(self.qwen_vl_interface).requires_grad_(False)
            ref_vlm.eval()
            object.__setattr__(self, "reference_qwen_vl", ref_vlm)
        else:
            logger.info("[QwenPIKM] KL disabled (kl_loss_weight=0); behaves identically to QwenPI.")
            object.__setattr__(self, "clap", None)
            object.__setattr__(self, "reference_qwen_vl", None)

    # ──────────────────────────────────────────────────────────────────
    # CLAP solutionizer: actions -> `<ACT_*>` token string
    # ──────────────────────────────────────────────────────────────────
    def _init_clap_solutionizer(self, clap_cfg) -> None:
        """Instantiate a frozen CLAP encoder used purely for action discretisation."""
        # Import lazily because the clap directory is only
        # available in repos that include it.
        try:
            from clap import modules as _clap_modules
            from clap.modules import ContrastiveDINOLatentActionModel
        except ImportError as e:
            raise ImportError(
                "QwenPIKM requires clap/modules/ContrastiveDINOLatentActionModel. "
                "Either provide the clap package or set kl_loss_weight=0."
            ) from e

        # CLAP's __init__ instantiates a DINO image encoder, but `action_vq_encode`
        # — the only path we need — never touches it. Stub the loader so we don't
        # have to ship DINO weights; the resulting Identity will still load_state_dict
        # cleanly because `dino_encoder.*` keys in the checkpoint will simply be
        # marked unexpected.
        original_loader = _clap_modules.clap.load_dino_encoder
        _clap_modules.clap.load_dino_encoder = lambda **_: nn.Identity()
        try:
            action_dim_per_arm = clap_cfg.get("action_dim_per_arm") or int(
                self.config.framework.action_model.action_dim
            )
            # CLAP was trained on a fixed chunk size (e.g. 32). The flow-matching
            # head's action_horizon may differ; we adapt action windows for CLAP
            # in `_actions_to_solutions`, but the encoder itself must be built
            # with the chunk length it was trained on.
            self.clap_chunk_size = int(clap_cfg.get("clap_chunk_size", 32))

            clap_module = ContrastiveDINOLatentActionModel(
                in_dim=int(clap_cfg.get("image_channels", 3)),
                model_dim=int(clap_cfg.get("model_dim", 768)),
                chunk_size=self.clap_chunk_size,
                latent_dim=int(clap_cfg.get("latent_dim", 128)),
                action_vae_dim=int(clap_cfg.get("action_vae_dim", 512)),
                max_action_dim=int(action_dim_per_arm),
                num_latents=int(clap_cfg.get("num_latents", 32)),
                num_t_codes=int(clap_cfg.get("num_t_codes", 8)),
                visual_t_codes=int(clap_cfg.get("visual_t_codes", 8)),
                patch_size=int(clap_cfg.get("patch_size", 16)),
                enc_blocks=int(clap_cfg.get("enc_blocks", 12)),
                dec_blocks=int(clap_cfg.get("dec_blocks", 12)),
                num_heads=int(clap_cfg.get("num_heads", 12)),
                dropout=float(clap_cfg.get("dropout", 0.0)),
                # Anything DINO-related is ignored by our stub.
                dino_model_type=str(clap_cfg.get("dino_model_type", "dinov2")),
                dino_model_variant=str(clap_cfg.get("dino_model_variant", "vits14")),
                dino_model_path=str(clap_cfg.get("dino_model_path", "facebook/dinov2-small")),
                dino_weights_path=clap_cfg.get("dino_weights_path", None),
            ).float()
        finally:
            _clap_modules.clap.load_dino_encoder = original_loader

        clap_ckpt_path = clap_cfg.get("clap_ckpt", None)
        if clap_ckpt_path is None:
            raise ValueError("framework.knowledge_matching.clap.clap_ckpt must be set")
        logger.info(f"[QwenPIKM] Loading frozen CLAP weights from {clap_ckpt_path}")
        ckpt = torch.load(clap_ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        # Strip the lightning "clap." prefix; drop pipeline modules that
        # belong to the trainer rather than to CLAP itself, and DINO weights
        # we don't need.
        sd = {k.replace("clap.", ""): v for k, v in sd.items()}
        sd = {
            k: v for k, v in sd.items()
            if "astribot_pipeline" not in k
            and "action_denormalization" not in k
            and not k.startswith("dino_encoder")
        }
        missing, unexpected = clap_module.load_state_dict(sd, strict=False)
        # Filter out DINO-only missing keys from the count for a clean log.
        non_dino_missing = [k for k in missing if not k.startswith("dino_encoder")]
        logger.info(
            f"[QwenPIKM] CLAP load_state_dict: missing={len(non_dino_missing)} "
            f"(plus DINO stubs) unexpected={len(unexpected)}"
        )
        clap_module.requires_grad_(False)
        clap_module.eval()
        # Bypass nn.Module's submodule registration so DeepSpeed's optimizer
        # doesn't see frozen CLAP params.
        object.__setattr__(self, "clap", clap_module)
        self.clap_num_action_codes = int(clap_cfg.get("num_t_codes", 8))

        # Resolve and validate ``dual_arm_indices`` now that CLAP exists. We
        # pre-build a long index tensor of length 2 * d_per_arm — left first,
        # right second — so ``_actions_to_solutions`` can do a single
        # ``index_select`` to permute the dataset's native column order into
        # the contiguous ``[L | R]`` layout CLAP was trained on.
        idx_cfg = clap_cfg.get("dual_arm_indices", None)
        self._km_dual_arm_perm: Optional[torch.Tensor] = None
        if idx_cfg is not None:
            if self.km_arm_layout != "dual":
                raise ValueError(
                    "framework.knowledge_matching.clap.dual_arm_indices is only meaningful "
                    f"when arm_layout='dual', got arm_layout={self.km_arm_layout!r}."
                )
            try:
                left_idx = list(idx_cfg["left"])
                right_idx = list(idx_cfg["right"])
            except (KeyError, TypeError) as e:
                raise ValueError(
                    "knowledge_matching.clap.dual_arm_indices must be a dict with "
                    "'left' and 'right' integer lists."
                ) from e
            d_per_arm = int(self.clap.max_action_dim)
            if len(left_idx) != d_per_arm or len(right_idx) != d_per_arm:
                raise ValueError(
                    f"dual_arm_indices: each arm needs exactly action_dim_per_arm={d_per_arm} "
                    f"entries, got left={len(left_idx)} right={len(right_idx)}."
                )
            self._km_dual_arm_perm = torch.tensor(
                left_idx + right_idx, dtype=torch.long
            )

    @torch.no_grad()
    def _actions_to_solutions(self, actions: torch.Tensor) -> List[str]:
        """Map [B, T, D] continuous actions to `<ACT_*>` token strings via frozen CLAP.

        CLAP only ever saw windows of length ``clap_chunk_size`` (e.g. 32) at
        training time, so we resample the action window to that length.
        The flow-matching policy itself keeps its native ``action_horizon``;
        only the KL-side discretiser is rescaled.

        Dual-arm note: when ``arm_layout == "dual"``, ``D`` must be ``2 *
        action_dim_per_arm``.  We split into left / right halves, encode each
        via CLAP, and concatenate the two ``<ACT_*>`` strings — same wire
        format as ``encode_action_to_tokens`` in
        ``clap/model_clap_vla_fm.py`` with the default
        arm_mask = [1, 1].
        """
        # Lazily move frozen CLAP/reference VLM to the live VLM's device on
        # first use. They live outside the nn.Module submodule tree so that
        # DeepSpeed's optimizer doesn't see them; that also means
        # `model.cuda()` does not move them.
        target_device = next(self.qwen_vl_interface.model.parameters()).device
        if next(self.clap.parameters()).device != target_device:
            self.clap.to(target_device)
        if self.reference_qwen_vl is not None and next(
            self.reference_qwen_vl.parameters()
        ).device != target_device:
            self.reference_qwen_vl.to(target_device)

        actions = actions.to(target_device)
        B, T, D = actions.shape
        target = self.clap_chunk_size
        if T == target:
            actions_clap = actions
        elif T > target:
            actions_clap = actions[:, -target:, :]
        else:
            pad_steps = target - T
            last = actions[:, -1:, :].expand(-1, pad_steps, -1)
            actions_clap = torch.cat([actions, last], dim=1)

        if self.km_arm_layout == "dual":
            # Encode left/right separately. CLAP was trained per-arm; the
            # assistant-turn token order must match CLAP-VLA-FM's legacy
            # Stage-3 format ("right_indices first, then left_indices") — see
            # ``format_action_token_strings`` in
            # ``clap/unified_action.py``. Mismatched order
            # would weaken the KL signal because the frozen reference VLM was
            # only ever asked to score right-first sequences.
            d_per_arm = self.clap.max_action_dim
            # Optional column permutation: dataset-native column order into the
            # contiguous ``[L | R]`` layout CLAP expects. Only the KL path uses
            # the permuted vector — the flow-matching loss & ``predict_action``
            # see the original layout, so eval/serving is unaffected.
            if self._km_dual_arm_perm is not None:
                perm = self._km_dual_arm_perm.to(actions_clap.device)
                if D != perm.numel():
                    raise ValueError(
                        f"[QwenPIKM] dual_arm_indices spans {perm.numel()} columns but "
                        f"action dim is {D}."
                    )
                actions_clap = actions_clap.index_select(-1, perm)
            elif D != 2 * d_per_arm:
                raise ValueError(
                    f"[QwenPIKM] arm_layout=dual expects action dim = 2*{d_per_arm}={2*d_per_arm}, "
                    f"got {D}. Set knowledge_matching.clap.dual_arm_indices when the dataset "
                    f"interleaves arms differently."
                )
            left = actions_clap[..., :d_per_arm].float()
            right = actions_clap[..., d_per_arm:2 * d_per_arm].float()
            with torch.autocast("cuda", enabled=False):
                left_idx = self.clap.action_vq_encode(left)["indices"]    # [B, num_t_codes]
                right_idx = self.clap.action_vq_encode(right)["indices"]
            left_np = left_idx.detach().cpu().numpy()
            right_np = right_idx.detach().cpu().numpy()
            return [
                "".join(f"<ACT_{int(i)}>" for i in right_np[b])
                + "".join(f"<ACT_{int(i)}>" for i in left_np[b])
                for b in range(B)
            ]

        with torch.autocast("cuda", enabled=False):
            outputs = self.clap.action_vq_encode(actions_clap.float())
            indices = outputs["indices"]  # [B, num_t_codes]
        idx_np = indices.detach().cpu().numpy()
        return ["".join(f"<ACT_{int(i)}>" for i in idx_np[b]) for b in range(idx_np.shape[0])]

    # ──────────────────────────────────────────────────────────────────
    # Override _encode_vl_hidden_states so BOTH the action-loss forward AND
    # `predict_action` use the same prompt format the reference VLM was
    # pretrained with. This matters because the student VLM is co-trained
    # with a KL regulariser against that reference; mismatched prompts at
    # train/inference vs KL would weaken the constraint AND produce a
    # train/test gap. The shared Qwen3 wrapper is left untouched so other
    # frameworks (QwenPI / QwenOFT / QwenFAST / QwenGR00T) keep their own
    # CoT_prompt-based behaviour.
    # ──────────────────────────────────────────────────────────────────
    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str], robot_types: Optional[List[str]] = None
    ) -> tuple:
        # Allow callers (notably ``QwenPI.forward`` / ``QwenPI.predict_action``,
        # which we inherit unchanged) to feed robot types via an instance stash
        # set by our overridden forward/predict_action.
        if robot_types is None:
            robot_types = getattr(self, "_pending_robot_types", None)
        qwen_inputs = self._build_qwen_inputs_no_solution(
            batch_images, instructions, robot_types=robot_types
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        # Knowledge Insulating: freeze the VLM at training time. The action
        # expert still gets gradients via its own parameters; the VLM forward
        # is just a feature extractor here.
        ki_grad_ctx = torch.no_grad() if self.enable_ki and self.training else _NullCtx()
        with ki_grad_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # `hidden_states` has length num_hidden_layers + 1 (index 0 is the
            # embedding output, k+1 is VLM block k's output).  Our
            # `vl_layer_indices` are 0-indexed over VLM blocks, so offset by 1.
            all_hidden = qwenvl_outputs.hidden_states
            vl_embs_list = [all_hidden[i + 1] for i in self.vl_layer_indices]
            assert len(vl_embs_list) == len(self.action_model.model.transformer_blocks), (
                f"vl_layer_indices ({len(vl_embs_list)}) must match DiT depth "
                f"({len(self.action_model.model.transformer_blocks)})."
            )
        return vl_embs_list, attention_mask

    def _build_qwen_inputs_no_solution(
        self,
        batch_images: List,
        instructions: List[str],
        robot_types: Optional[List[str]] = None,
        robot_type: str = "franka",   # legacy single-string fallback
    ) -> dict:
        """Build inference-time inputs (no assistant turn) using the
        CLAP-VLA-FM prompt template. Same wording as
        ``_build_clap_vla_fm_qwen_inputs`` but with
        ``add_generation_prompt=True`` and no solution.

        ``robot_types`` (per-sample) takes precedence over ``robot_type``
        (single-string default). Pass per-sample types when training on
        Astribot/AgiBot/etc. so the VLM sees the same "Robot category: ..."
        line CLAP-VLA-FM was trained with."""
        processor = self.qwen_vl_interface.processor
        if robot_types is None:
            robot_types = [robot_type] * len(instructions)
        if len(robot_types) != len(instructions):
            raise ValueError(
                f"robot_types length {len(robot_types)} != instructions length {len(instructions)}"
            )
        messages = []
        for imgs, instruction, rt in zip(batch_images, instructions, robot_types):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({
                "type": "text",
                "text": (
                    f"Robot category: {rt}\n"
                    f"Control the robot to do the task: {instruction}\n"
                    "Please output the subtask and the action tokens."
                ),
            })
            messages.append([
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": content},
            ])
        batch_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.qwen_vl_interface.model.device)

    # ──────────────────────────────────────────────────────────────────
    # KL forward pass
    # ──────────────────────────────────────────────────────────────────
    def _build_clap_vla_fm_qwen_inputs(
        self,
        batch_images: List,
        instructions: List[str],
        solutions: List[str],
        robot_types: Optional[List[str]] = None,
        robot_type: str = "franka",   # legacy single-string fallback
    ) -> dict:
        """Replicate the prompt/assistant template used by CLAP-VLA-FM exactly.

        This is intentionally NOT delegated to ``self.qwen_vl_interface.
        build_qwenvl_inputs`` because that method applies the starVLA
        ``CoT_prompt`` template, which differs from the format the frozen
        reference VLM was trained with. Mismatching formats would weaken the
        KL signal.

        Mirrors ``CLAP_VLA_FM.build_qwenvl_inputs`` in
        ``clap/model_clap_vla_fm.py``:
          - system  : "You are a helpful assistant."
          - user    : images + "Robot category: {robot_type}\\nControl the
                       robot to do the task: {instruction}\\nPlease output
                       the subtask and the action tokens."
          - assistant: "Subtask: {subtask}\\nAction: {action_tokens}"
            (we drop subtask labels because they are not part of the LIBERO
            dataset; the assistant turn is just the solution string passed in.)
        """
        processor = self.qwen_vl_interface.processor
        if robot_types is None:
            robot_types = [robot_type] * len(instructions)
        if len(robot_types) != len(instructions):
            raise ValueError(
                f"robot_types length {len(robot_types)} != instructions length {len(instructions)}"
            )
        messages = []
        for imgs, instruction, solution, rt in zip(
            batch_images, instructions, solutions, robot_types
        ):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({
                "type": "text",
                "text": (
                    f"Robot category: {rt}\n"
                    f"Control the robot to do the task: {instruction}\n"
                    "Please output the subtask and the action tokens."
                ),
            })
            messages.append([
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": solution}]},
            ])

        batch_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # Build labels so only action-token positions contribute to the KL.
        # Strategy mirrors the starVLA Qwen3 wrapper: anything before the
        # first <ACT_*> id is IGNORE_INDEX; padding is also masked out.
        labels = batch_inputs["input_ids"].clone()
        act_min, act_max = self._action_token_id_range()
        for b in range(labels.size(0)):
            seq = labels[b]
            in_range = (seq >= act_min) & (seq <= act_max)
            nz = torch.nonzero(in_range, as_tuple=False)
            if nz.numel() > 0:
                labels[b, : nz[0].item()] = IGNORE_INDEX
            else:
                labels[b, :] = IGNORE_INDEX
        labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch_inputs["labels"] = labels
        return batch_inputs.to(self.qwen_vl_interface.model.device)

    def _action_token_id_range(self) -> Tuple[int, int]:
        """Return (min, max) inclusive token ids of the `<ACT_*>` range.

        Falls back to the constants defined in
        ``starVLA/model/modules/vlm/QWen3.py`` (``_ACTION_TOKEN_MIN/MAX``) so
        we don't have to retokenize the action vocabulary every step.
        """
        if hasattr(self.qwen_vl_interface, "_ACTION_TOKEN_MIN") and hasattr(
            self.qwen_vl_interface, "_ACTION_TOKEN_MAX"
        ):
            return int(self.qwen_vl_interface._ACTION_TOKEN_MIN), int(
                self.qwen_vl_interface._ACTION_TOKEN_MAX
            )
        from starVLA.model.modules.vlm.QWen3 import _ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX
        return int(_ACTION_TOKEN_MIN), int(_ACTION_TOKEN_MAX)

    def _knowledge_matching_kl_loss(
        self,
        batch_images: List,
        instructions: List[str],
        actions: torch.Tensor,
        robot_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """KL(student, teacher) averaged over GT-action-token positions.

        Mirrors clap/model_clap_vla_fm.py:
          1. Discretise GT actions into a `<ACT_*>` sequence with frozen CLAP.
             CLAP expects its training chunk length (e.g. 32);
             ``_actions_to_solutions`` reshapes the window if the
             flow-matching action_horizon differs.
          2. Build qwen_inputs with that sequence as the assistant turn,
             using *exactly* the same prompt template the reference VLM was
             trained with — see ``_build_clap_vla_fm_qwen_inputs``.
          3. Forward both the trainable VLM and the frozen reference VLM.
          4. KL is averaged over `labels != IGNORE_INDEX`, i.e. the GT
             action-token positions.
        """
        token_strings = self._actions_to_solutions(actions)
        # Default subtask is empty for LIBERO; format string keeps the KL
        # focused on the `<ACT_*>` tail of the assistant message.
        solutions = [self.solution_template.format(action_tokens=s) for s in token_strings]

        qwen_inputs = self._build_clap_vla_fm_qwen_inputs(
            batch_images=batch_images,
            instructions=instructions,
            solutions=solutions,
            robot_types=robot_types,
        )
        labels = qwen_inputs["labels"]

        # Student forward (gradients ON).
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_out = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        # Teacher forward (frozen, no grads).
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_out = self.reference_qwen_vl(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        student_logp = F.log_softmax(student_out.logits.to(torch.float32), dim=-1)
        teacher_logp = F.log_softmax(teacher_out.logits.to(torch.float32), dim=-1)

        # F.kl_div(input=log_p, target=log_q, log_target=True)
        #   = sum_x exp(log_q) * (log_q - log_p) per element = KL(q || p).
        if self.kl_type == "kl":
            # Forward KL: KL(student || teacher).  q=student, p=teacher.
            per_token_kl = F.kl_div(
                teacher_logp, student_logp, reduction="none", log_target=True
            ).sum(dim=-1)
        else:  # "reverse_kl"
            # Reverse KL: KL(teacher || student).  q=teacher, p=student.
            # CLAP-VLA-FM uses this variant.
            per_token_kl = F.kl_div(
                student_logp, teacher_logp, reduction="none", log_target=True
            ).sum(dim=-1)

        mask = labels != IGNORE_INDEX
        if mask.any():
            kl_loss = per_token_kl[mask].mean()
        else:
            # Defensive: never break the optimizer with a NaN / disconnected scalar.
            kl_loss = per_token_kl.sum() * 0.0
        return kl_loss

    # ──────────────────────────────────────────────────────────────────
    # forward = QwenPI.forward + KL
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_robot_types(examples: List[dict]) -> Optional[List[str]]:
        """Pull a per-sample ``robot_type`` list out of examples (if present)."""
        if examples is None or len(examples) == 0:
            return None
        if "robot_type" not in examples[0]:
            return None
        return [str(ex.get("robot_type", "franka")) for ex in examples]

    def forward(self, examples: List[dict] = None, **kwargs):
        # Stash per-sample robot types so the inherited QwenPI.forward, which
        # calls ``_encode_vl_hidden_states`` without knowing about robot types,
        # picks them up via our override.
        robot_types = self._extract_robot_types(examples)
        self._pending_robot_types = robot_types
        try:
            out = super().forward(examples=examples, **kwargs)
        finally:
            self._pending_robot_types = None
        action_loss = out["action_loss"]

        if self.kl_loss_weight > 0 and self.reference_qwen_vl is not None:
            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]
            actions_list = [example["action"] for example in examples]
            actions = torch.tensor(np.array(actions_list), device=action_loss.device)

            kl_loss = self._knowledge_matching_kl_loss(
                batch_images, instructions, actions, robot_types=robot_types
            )
            return {
                "action_loss": action_loss + self.kl_loss_weight * kl_loss,
                "kl_loss": kl_loss.detach(),
                "fm_loss": action_loss.detach(),
            }
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        # Same robot_type plumbing as forward, but for inference.
        robot_types = self._extract_robot_types(examples) if isinstance(examples, list) else None
        self._pending_robot_types = robot_types
        try:
            return super().predict_action(examples=examples, **kwargs)
        finally:
            self._pending_robot_types = None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke test:  python starVLA/model/framework/VLM4A/QwenPIKM.py \
#                            --config_yaml examples/Astribot/train_files/...yaml \
#                            [--enable_ki] [--kl_loss_weight 0]
# Builds the framework on a single GPU (or CPU fallback), runs one forward
# with two synthetic Astribot-shaped examples, and verifies that the
# resulting action_loss is finite. KL/KI paths are exercised via flags.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--kl_loss_weight", type=float, default=None,
                        help="Override framework.knowledge_matching.kl_loss_weight")
    parser.add_argument("--enable_ki", action="store_true",
                        help="Force framework.enable_ki=True (sets kl_loss_weight=0).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    if args.enable_ki:
        cfg.framework.enable_ki = True
        cfg.framework.knowledge_matching.kl_loss_weight = 0.0
    elif args.kl_loss_weight is not None:
        cfg.framework.knowledge_matching.kl_loss_weight = args.kl_loss_weight

    model = Qwen_PI_KM(cfg).to(args.device)
    model.train()
    action_dim = int(cfg.framework.action_model.action_dim)
    horizon = int(cfg.framework.action_model.action_horizon)
    state_dim = int(cfg.framework.action_model.state_dim)

    # Two minimal Astribot-shaped examples (3 cameras, dual-arm 14-dim).
    img = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
    sample = {
        "image": [img, img, img],
        "lang": "put the doll in the basket",
        "action": np.random.uniform(-1, 1, size=(horizon, action_dim)).astype(np.float32),
        "state": np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float32),
        "robot_type": "S1-stationary",
    }
    out = model.forward(examples=[sample, sample])
    print("[QwenPIKM smoke]", {k: float(v) for k, v in out.items() if hasattr(v, "item")})
    print("[QwenPIKM smoke] OK")

