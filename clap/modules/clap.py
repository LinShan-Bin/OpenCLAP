import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from clap.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer, ResidualVectorQuantizer

# ActionEmbed, TimestepEmbed, ActionEncoder, ActionDecoder no longer needed - replaced by MldVae
from clap.modules.mld_vae import MldVae
from clap.modules.gather import Gather
from clap.unified_action import (
    flatten_active_arm_actions,
    get_arm_mask,
)


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_dino_encoder(dino_model_type: str, dino_model_variant: str, 
                     dino_model_path: str = None, dino_weights_path: str = None):
    """Load DINO encoder based on configuration."""
    if dino_model_type == "dinov2":
        # Load DINOv2 from torch hub
        encoder = torch.hub.load(dino_model_path, dino_model_variant)
    elif dino_model_type == "dinov3":
        # Load DINOv3 from local path
        encoder = torch.hub.load(
            dino_model_path,
            dino_model_variant,
            source='local',
            weights=dino_weights_path
        )
    else:
        raise ValueError(f"Unsupported DINO model type: {dino_model_type}")
    
    encoder.requires_grad_(False)
    return encoder


class ContrastiveDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            action_vae_dim: int,
            max_action_dim: int,
            num_t_codes: int,
            chunk_size: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            visual_t_codes: int = 8,
            use_residual_vq: bool = False,
            action_layers: int = 15,
            dropout: float = 0.0,
            dino_model_type: str = "dinov3",
            dino_model_variant: str = "dinov3_vitb16",
            dino_model_path: str = "./pretrained/dinov3",
            dino_weights_path: str = "./pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ) -> None:
        super(ContrastiveDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.image_aug_brightness = 0.2
        self.image_aug_contrast = 0.2
        self.image_aug_grayscale_prob = 0.1
        self.image_aug_noise_std = 0.02
        self.proxy_ema_decay = 0.999
        # Load DINO encoder based on configuration
        self.dino_encoder = load_dino_encoder(
            dino_model_type=dino_model_type,
            dino_model_variant=dino_model_variant,
            dino_model_path=dino_model_path,
            dino_weights_path=dino_weights_path
        )

        dino_dim = 768

        self.num_i_codes = 2
        self.num_t_codes = num_t_codes
        self.visual_t_codes = visual_t_codes
        if use_residual_vq:
            self.num_i_codes = 1
            self.num_t_codes = 1  # Use residual vector quantizer for action VQ-VAE
        self.num_codes = self.num_i_codes + self.visual_t_codes * 2
        self.action_latent_i = nn.Parameter(torch.empty(1, 1, self.num_i_codes, dino_dim))
        self.action_latent_t = nn.Parameter(torch.empty(1, 1, self.visual_t_codes * 2, dino_dim))
        nn.init.uniform_(self.action_latent_i, a=-1, b=1)
        nn.init.uniform_(self.action_latent_t, a=-1, b=1)
        self.visual_encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        # self.to_codebook_i = nn.Sequential(
        #     nn.Linear(model_dim, latent_dim),
        #     nn.LayerNorm(latent_dim)
        # )
        # self.to_codebook_t = nn.Sequential(
        #     nn.Linear(model_dim, latent_dim),
        #     nn.LayerNorm(latent_dim)
        # )
        self.to_codebook_i = nn.Linear(model_dim, latent_dim)  # Since penalization is applied to i-codes, we don't need layer norm
        self.to_codebook_t = nn.Linear(model_dim, latent_dim)
        self._init_ema_visual_encoder()
        
        if use_residual_vq:
            self.vq_i = ResidualVectorQuantizer(
                n_codebooks=2,
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
            self.vq_t = ResidualVectorQuantizer(
                n_codebooks=num_t_codes,
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        else:
            self.vq_i = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
                # norm_init=True,
            )
            self.vq_t = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
                # norm_init=True,
            )  # Shared
        self.vq_t.requires_grad_(False)
        
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.visual_decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Create a simple ablation config for MldVae
        class SimpleAblation:
            MLP_DIST = True
            PE_TYPE = "mld"
        
        # Initialize MldVae for action encoding/decoding
        self.action_vae = MldVae(
            ablation=SimpleAblation(),
            nfeats=max_action_dim,  # Input/output feature dimension
            latent_dim=[self.num_t_codes, action_vae_dim],  # [latent_size, action_vae_dim]
            codebook_dim=latent_dim,
            ff_size=1024,
            num_layers=action_layers,
            num_heads=4,
            dropout=dropout,
            arch="all_encoder",  # Use all_encoder architecture
            normalize_before=False,
            activation="gelu",
            position_embedding="learned"
        )
        self.action_vae.requires_grad_(False)
        
        # SigLIP parameters
        self.temperature = nn.Parameter(torch.log(torch.tensor(10.0)))  # Learnable temperature (log-parameterized)
        self.bias = nn.Parameter(torch.tensor(-10.0))  # Learnable bias
    
    def random_restart(self) -> None:
        self.vq_i.random_restart()
        self.vq_i.reset_usage()

    def _init_ema_visual_encoder(self) -> None:
        self.ema_visual_encoder = copy.deepcopy(self.visual_encoder)
        self.ema_to_codebook_i = copy.deepcopy(self.to_codebook_i)
        self.ema_to_codebook_t = copy.deepcopy(self.to_codebook_t)
        self.register_buffer("ema_action_latent_i", self.action_latent_i.detach().clone())
        self.register_buffer("ema_action_latent_t", self.action_latent_t.detach().clone())
        self.reset_ema_encoder()

    def _set_ema_encoder_eval(self) -> None:
        self.ema_visual_encoder.requires_grad_(False)
        self.ema_to_codebook_i.requires_grad_(False)
        self.ema_to_codebook_t.requires_grad_(False)
        self.ema_visual_encoder.eval()
        self.ema_to_codebook_i.eval()
        self.ema_to_codebook_t.eval()

    @torch.no_grad()
    def reset_ema_encoder(self) -> None:
        self.ema_visual_encoder.load_state_dict(self.visual_encoder.state_dict())
        self.ema_to_codebook_i.load_state_dict(self.to_codebook_i.state_dict())
        self.ema_to_codebook_t.load_state_dict(self.to_codebook_t.state_dict())
        self.ema_action_latent_i.copy_(self.action_latent_i.detach())
        self.ema_action_latent_t.copy_(self.action_latent_t.detach())
        self._set_ema_encoder_eval()

    @torch.no_grad()
    def update_ema_encoder(self) -> None:
        decay = self.proxy_ema_decay
        self.ema_action_latent_i.mul_(decay).add_(self.action_latent_i.detach(), alpha=1 - decay)
        self.ema_action_latent_t.mul_(decay).add_(self.action_latent_t.detach(), alpha=1 - decay)

        for ema_module, online_module in (
            (self.ema_visual_encoder, self.visual_encoder),
            (self.ema_to_codebook_i, self.to_codebook_i),
            (self.ema_to_codebook_t, self.to_codebook_t),
        ):
            for ema_param, online_param in zip(ema_module.parameters(), online_module.parameters()):
                ema_param.mul_(decay).add_(online_param.detach(), alpha=1 - decay)
            for ema_buffer, online_buffer in zip(ema_module.buffers(), online_module.buffers()):
                if torch.is_floating_point(ema_buffer):
                    ema_buffer.mul_(decay).add_(online_buffer.detach(), alpha=1 - decay)
                else:
                    ema_buffer.copy_(online_buffer)
        self._set_ema_encoder_eval()

    def augment_videos(self, videos: Tensor) -> Tensor:
        if not self.training:
            return videos

        B = videos.shape[0]
        device = videos.device
        dtype = videos.dtype
        videos = videos.clamp(0, 1)
        sample_shape = (B, 1, 1, 1, 1)

        brightness = 1 + (
            torch.rand(sample_shape, device=device, dtype=dtype) * 2 - 1
        ) * self.image_aug_brightness
        videos = videos * brightness

        contrast = 1 + (
            torch.rand(sample_shape, device=device, dtype=dtype) * 2 - 1
        ) * self.image_aug_contrast
        mean = videos.mean(dim=(-3, -2, -1), keepdim=True)
        videos = (videos - mean) * contrast + mean

        grayscale_mask = (
            torch.rand(sample_shape, device=device) < self.image_aug_grayscale_prob
        )
        gray = videos.mean(dim=-3, keepdim=True).expand_as(videos)
        videos = torch.where(grayscale_mask, gray, videos)

        if self.image_aug_noise_std > 0:
            videos = videos + torch.randn_like(videos) * self.image_aug_noise_std

        return videos.clamp(0, 1)

    def t_code_features(self, outputs: Dict) -> Tensor:
        features = outputs["z_q"][:, 0, self.num_i_codes:self.num_codes]
        return features.reshape(-1, 2, self.latent_dim * self.visual_t_codes)
    
    def compute_siglip_loss(self, visual_features: Tensor, action_features: Tensor) -> Tensor:
        """
        Compute SigLIP contrastive loss between visual and action features.
        Args:
            visual_features: [B, contrastive_dim] - Before L2 normalization
            action_features: [B, contrastive_dim] - Before L2 normalization
        Returns:
            SigLIP contrastive loss
        """
        # visual_features = F.normalize(self.visual_proj(visual_features), dim=1)
        # action_features = F.normalize(self.action_proj(action_features), dim=1)
        visual_features = F.normalize(visual_features, dim=1)
        action_features = F.normalize(action_features, dim=1)

        # Gather features from all GPUs for distributed training        
        if torch.distributed.is_initialized():
            all_visual_features = Gather(visual_features)  # [B*world_size, contrastive_dim]
            all_action_features = Gather(action_features)   # [B*world_size, contrastive_dim]
            
            # Get current GPU info
            rank = torch.distributed.get_rank()
            bs = visual_features.shape[0]
            
            # Compute logits for current GPU's batch vs all gathered features
            logits_visual_to_action = torch.exp(self.temperature) * (
                visual_features @ all_action_features.t()) + self.bias  # [B, B*world_size]
            logits_action_to_visual = torch.exp(self.temperature) * (
                action_features @ all_visual_features.t()) + self.bias  # [B, B*world_size]
            
            # Create labels: positive pairs are on the diagonal for current rank's offset
            total_bs = all_visual_features.shape[0]
            all_labels = torch.eye(total_bs, device=visual_features.device)
            labels_visual_to_action = all_labels[rank * bs : (rank + 1) * bs]
            labels_action_to_visual = all_labels[rank * bs : (rank + 1) * bs]
        else:
            # Single GPU case
            logits_visual_to_action = torch.exp(self.temperature) * (
                visual_features @ action_features.t()) + self.bias  # [B, B]
            logits_action_to_visual = torch.exp(self.temperature) * (
                action_features @ visual_features.t()) + self.bias  # [B, B]
            
            # Create labels: positive pairs are on the diagonal
            bs = visual_features.shape[0]
            labels_visual_to_action = torch.eye(bs, device=visual_features.device)  # [B, B]
            labels_action_to_visual = torch.eye(bs, device=visual_features.device)  # [B, B]
        
        # Convert binary labels to +1/-1 format for SigLIP
        labels_visual_to_action = 2 * labels_visual_to_action - 1  # +1 for positive, -1 for negative
        labels_action_to_visual = 2 * labels_action_to_visual - 1  # +1 for positive, -1 for negative
        
        # Compute SigLIP loss: -log(sigmoid(labels * logits)) = softplus(-labels * logits)
        loss_visual_to_action = F.softplus(-labels_visual_to_action * logits_visual_to_action).sum() / bs
        loss_action_to_visual = F.softplus(-labels_action_to_visual * logits_action_to_visual).sum() / bs
        
        # Total contrastive loss (symmetric)
        siglip_loss = (loss_visual_to_action + loss_action_to_visual) / 2
        return siglip_loss
    
    def visual_vq_encode(
        self,
        videos: Tensor,
        lang_embed: Tensor = None,
        augment: bool = True,
        use_ema_encoder: bool = False,
    ) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        if augment:
            videos = self.augment_videos(videos)
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        if use_ema_encoder:
            self._set_ema_encoder_eval()
            action_latent_i = self.ema_action_latent_i
            action_latent_t = self.ema_action_latent_t
            visual_encoder = self.ema_visual_encoder
            to_codebook_i = self.ema_to_codebook_i
            to_codebook_t = self.ema_to_codebook_t
        else:
            action_latent_i = self.action_latent_i
            action_latent_t = self.action_latent_t
            visual_encoder = self.visual_encoder
            to_codebook_i = self.to_codebook_i
            to_codebook_t = self.to_codebook_t

        action_pad_i = action_latent_i.expand(B, T, -1, -1)
        action_pad_t = action_latent_t.expand(B, T, -1, -1)
        action_pad = torch.cat([action_pad_i, action_pad_t], dim=2)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = visual_encoder(padded_patches)

        # Get latent action for all future frames
        z_i = z[:, 1:, :self.num_i_codes]  # (B, T-1, n_i, E)
        z_t = z[:, 1:, self.num_i_codes: self.num_codes]  # (B, T-1, n_t, E)
        
        z_i = to_codebook_i(z_i)
        z_t = to_codebook_t(z_t)

        # Vector quantize separately for i and t
        z_i_flat = z_i.reshape(B * (T - 1), self.num_i_codes, self.latent_dim)
        z_t_flat = z_t.reshape(B * (T - 1), self.visual_t_codes * 2, self.latent_dim)
        
        z_q_i, z_orig_i, emb_i, indices_i = self.vq_i(z_i_flat)
        z_q_t, z_orig_t, emb_t, indices_t = self.vq_t(z_t_flat)
        
        z_q_i = z_q_i.reshape(B, T - 1, self.num_i_codes, self.latent_dim)
        z_q_t = z_q_t.reshape(B, T - 1, self.visual_t_codes * 2, self.latent_dim)
        
        # Concatenate the quantized results
        z_q = torch.cat([z_q_i, z_q_t], dim=2)
        z = torch.cat([z_orig_i, z_orig_t], dim=1)  # Concatenate for loss computation
        emb = torch.cat([emb_i, emb_t], dim=1)
        indices = torch.cat([indices_i, indices_t], dim=1)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }

    def forward(self, batch: Dict) -> Dict:
        # lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
        # lang_embed = self.lang_proj(lang_embed)

        outputs = self.visual_vq_encode(videos=batch["observation.head"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.visual_decoder(x=video_action_patches)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]]

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        
        # Prepare visual features for SigLIP
        visual_features_i = outputs["z_q"][:, 0, :self.num_i_codes]

        visual_features = self.t_code_features(outputs)
        
        action = batch["action"]
        robot_id = batch["robot_id"]
        if not isinstance(robot_id, torch.Tensor):
            robot_id = torch.tensor(robot_id, device=action.device)
        robot_id = robot_id.to(device=action.device).view(-1)
        non_contrastive_mask = robot_id == 2  # data without action labels
        # Prepare action features for SigLIP
        with torch.no_grad():
            Bc = action.shape[0]
            arm_mask = get_arm_mask(
                batch,
                Bc,
                default_layout="dual",
                device=action.device,
                dtype=torch.bool,
            )
            action_outputs = None
            action_arm_features = visual_features.new_zeros(
                Bc, 2, self.visual_t_codes, self.latent_dim
            )
            robot_mask = ~non_contrastive_mask
            if robot_mask.any():
                robot_arm_mask = arm_mask[robot_mask]
                active_action = flatten_active_arm_actions(
                    action[robot_mask],
                    robot_arm_mask,
                    self.max_action_dim,
                )
                action_outputs = self.action_vq_encode(active_action)
                robot_indices = torch.nonzero(robot_mask, as_tuple=False).flatten()
                n_left = int(robot_arm_mask[:, 0].sum().item())
                if n_left > 0:
                    left_indices = robot_indices[robot_arm_mask[:, 0]]
                    action_arm_features[left_indices, 0] = action_outputs["z_q"][:n_left, :self.visual_t_codes]
                if robot_arm_mask[:, 1].any():
                    right_indices = robot_indices[robot_arm_mask[:, 1]]
                    action_arm_features[right_indices, 1] = action_outputs["z_q"][n_left:, :self.visual_t_codes]
            # action_features = action_features.reshape(-1, self.latent_dim * self.visual_t_codes * 2)
            action_features = action_arm_features.reshape(-1, 2, self.latent_dim * self.visual_t_codes)
            # Human videos do not have action labels, so use a clean EMA visual
            # pass as a detached action-side proxy instead of self-pairing.
            if non_contrastive_mask.any():
                human_outputs = self.visual_vq_encode(
                    batch["observation.head"][non_contrastive_mask],
                    augment=False,
                    use_ema_encoder=True,
                )
                action_features[non_contrastive_mask] = self.t_code_features(human_outputs)
        
        # Compute SigLIP loss
        # Left arm and right arm are separate sequences
        visual_features = visual_features[arm_mask]
        action_features = action_features[arm_mask]
        siglip_loss = self.compute_siglip_loss(visual_features, action_features)
        # siglip_loss = F.mse_loss(visual_features, action_features, reduction="mean")
        outputs["loss_siglip"] = siglip_loss
        outputs["reg_loss"] = visual_features_i.abs().mean()
        # outputs["z"][:, self.num_i_codes: self.num_codes] = action_features.reshape(-1, self.visual_t_codes * 2, self.latent_dim)
        outputs["action_outputs"] = action_outputs
        return outputs
    
    def action_vq_encode(self, action: Tensor) -> Dict:
        B, T, D = action.shape
        
        # Use MldVae encoder to get latent representation directly from action
        # Use act_lens if provided, otherwise assume all sequences have length T
        lengths = [T] * B
        z, _ = self.action_vae.encode(action, lengths)  # z: [latent_size, B, latent_dim], dist not needed
        
        # Reshape z to match expected format [B, num_t_codes, latent_dim]
        z = z.permute(1, 0, 2)  # [B, latent_size, latent_dim]

        # Vector quantize
        z_q, z_orig, emb, indices = self.vq_t(z)
        return {
            "z_q": z_q,
            "z": z_orig,
            "emb": emb,
            "indices": indices,
        }
    
    def forward_action(self, batch: Dict) -> Dict:
        # Action VQ-VAE
        action = batch["action"]
        arm_mask = get_arm_mask(
            batch,
            action.shape[0],
            default_layout="dual",
            device=action.device,
            dtype=torch.bool,
        )
        action = flatten_active_arm_actions(action, arm_mask, self.max_action_dim)
        B2, T, D = action.shape
        # Use action_vq_encode with proper act_lens (timesteps handled internally by MldVae)
        outputs = self.action_vq_encode(action)
        
        # Use MldVae decoder to reconstruct action
        # Prepare z_q for decoder: [latent_size, B, latent_dim]
        z_q_decode = outputs['z_q'].permute(1, 0, 2)  # [latent_size, B, latent_dim]
        
        lengths = [T] * B2
        action_recon = self.action_vae.decode(z_q_decode, lengths)  # [B, T, latent_dim]
        
        outputs.update(
            {
                "recon": action_recon,
                "target": action
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
    
    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None, action: Tensor = None, act_lens: Tensor = None) -> Dict:
        outputs = self.visual_vq_encode(videos, lang_embed)
        t_outputs = {
            "z_q": outputs["z_q"][:, :, self.num_i_codes:],
            "z": outputs["z"][:, self.num_i_codes:],
            "emb": outputs["emb"][:, self.num_i_codes:],
            "indices": outputs["indices"][:, self.num_i_codes:],
        }
        return t_outputs


class LatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            action_vae_dim: int,
            max_action_dim: int,
            chunk_size: int,
            num_latents: int,
            num_t_codes: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            visual_t_codes: int = 8,
            use_residual_vq: bool = False,
            action_layers: int = 15,
            dropout: float = 0.0,
            dino_model_type: str = "dinov3",
            dino_model_variant: str = "dinov3_vitb16",
            dino_model_path: str = "./pretrained/dinov3",
            dino_weights_path: str = "./pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            variable_length_training: bool = False
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.variable_length_training = variable_length_training
        
        self.num_t_codes = num_t_codes

        self.use_residual_vq = use_residual_vq
        if use_residual_vq:
            self.vq_t = ResidualVectorQuantizer(
                n_codebooks=num_t_codes,
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
                # norm_init=True,
            )
            self.num_t_codes = 1  # Use residual vector quantizer for action VQ-VAE
        else:
            self.vq_t = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
                # norm_init=True,
            )  # Shared
        
        # Create a simple ablation config for MldVae
        class SimpleAblation:
            MLP_DIST = True
            PE_TYPE = "mld"
        
        # Initialize MldVae for action encoding/decoding
        self.action_vae = MldVae(
            ablation=SimpleAblation(),
            nfeats=max_action_dim,  # Input/output feature dimension
            latent_dim=[self.num_t_codes, action_vae_dim],  # [latent_size, action_vae_dim]
            codebook_dim=latent_dim,
            ff_size=1024,
            num_layers=action_layers,
            num_heads=4,
            dropout=dropout,
            arch="all_encoder",  # Use all_encoder architecture
            normalize_before=False,
            activation="gelu",
            position_embedding="learned"
        )
    
    def random_restart(self) -> None:
        self.vq_t.random_restart()
        self.vq_t.reset_usage()
        
    def action_vq_encode(self, action: Tensor, act_lens: Tensor = None) -> Dict:
        B, T, D = action.shape
        
        # Use MldVae encoder to get latent representation directly from action
        # Use act_lens if provided, otherwise assume all sequences have length T
        if act_lens is not None:
            lengths = act_lens.cpu().tolist()  # Convert tensor to list of integers
        else:
            lengths = [T] * B
        z, _ = self.action_vae.encode(action, lengths)  # z: [latent_size, B, latent_dim], dist not needed
        
        # Reshape z to match expected format [B, num_t_codes, latent_dim]
        z = z.permute(1, 0, 2)  # [B, latent_size, latent_dim]

        # Vector quantize
        z_q, z_orig, emb, indices = self.vq_t(z)
        return {
            "z_q": z_q,
            "z": z_orig,
            "emb": emb,
            "indices": indices,
        }
    
    def forward(self, batch: Dict) -> Dict:
        # Action VQ-VAE
        action = batch["action"]
        arm_mask = get_arm_mask(
            batch,
            action.shape[0],
            default_layout="dual",
            device=action.device,
            dtype=torch.bool,
        )
        # For action VQ-VAE, we treat active left and right arms as separate sequences.
        action = flatten_active_arm_actions(action, arm_mask, self.max_action_dim)
        B2, T, D = action.shape  # [2*B, T, max_action_dim]
        
        # Use action_vq_encode with proper act_lens (timesteps handled internally by MldVae)
        outputs = self.action_vq_encode(action)
        
        # Convert act_lens to proper lengths list for MldVae decoder
        lengths = [T] * B2
        
        # Use MldVae decoder to reconstruct action
        # Prepare z_q for decoder: [latent_size, B, latent_dim]
        z_q_decode = outputs['z_q'].permute(1, 0, 2)  # [latent_size, B, latent_dim]
        
        action_recon = self.action_vae.decode(z_q_decode, lengths)  # [B, T, latent_dim]
        
        if self.variable_length_training:
            z_len = torch.randint(4, self.num_t_codes, (1,)).item()
            z_q_cut = z_q_decode[:z_len]
            action_recon_cut = self.action_vae.decode(z_q_cut, lengths)  # [B, T, latent_dim]
            loss_var_len = F.mse_loss(action_recon_cut, action)
            outputs["loss_var_len"] = loss_var_len
        
        outputs.update(
            {
                "recon": action_recon,
                "target": action
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class DualBranchLatentActionModel(nn.Module):
    """
    Dual-branch latent action model using supervised learning instead of contrastive learning.
    This serves as an ablation study for ContrastiveDINOLatentActionModel.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            max_action_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            dino_model_type: str = "dinov3",
            dino_model_variant: str = "dinov3_vitb16",
            dino_model_path: str = "./pretrained/dinov3",
            dino_weights_path: str = "./pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    ) -> None:
        super(DualBranchLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2
        self.max_action_dim = max_action_dim

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # Load DINO encoder based on configuration
        self.dino_encoder = load_dino_encoder(
            dino_model_type=dino_model_type,
            dino_model_variant=dino_model_variant,
            dino_model_path=dino_model_path,
            dino_weights_path=dino_weights_path
        )

        dino_dim = 768

        self.num_i_codes = 2
        self.num_t_codes = 4
        self.num_codes = self.num_i_codes + self.num_t_codes
        self.action_latent_i = nn.Parameter(torch.empty(1, 1, self.num_i_codes, dino_dim))
        self.action_latent_t = nn.Parameter(torch.empty(1, 1, self.num_t_codes, dino_dim))
        nn.init.uniform_(self.action_latent_i, a=-1, b=1)
        nn.init.uniform_(self.action_latent_t, a=-1, b=1)
        self.visual_encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook_i = nn.Linear(model_dim, latent_dim)
        self.to_codebook_t = nn.Linear(model_dim, latent_dim)
        self.vq_i = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        self.vq_t = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )  # Shared
        self.vq_t.requires_grad_(False)
        
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.visual_decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Create a simple ablation config for MldVae
        class SimpleAblation:
            MLP_DIST = True
            PE_TYPE = "mld"
        
        # Initialize MldVae for action encoding/decoding
        self.action_vae = MldVae(
            ablation=SimpleAblation(),
            nfeats=max_action_dim,  # Input/output feature dimension
            latent_dim=[self.num_t_codes, latent_dim],  # [latent_size, latent_dim]
            ff_size=1024,
            num_layers=9,
            num_heads=4,
            dropout=dropout,
            arch="all_encoder",  # Use all_encoder architecture
            normalize_before=False,
            activation="gelu",
            position_embedding="learned"
        )
        self.action_vae.requires_grad_(False)
        
        # Action decoder for supervised learning
        # Decode visual features (from t-codes) to action space
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim * self.num_t_codes, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, max_action_dim)
        )
    
    def visual_vq_encode(self, videos: Tensor, lang_embed: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad_i = self.action_latent_i.expand(B, T, -1, -1)
        action_pad_t = self.action_latent_t.expand(B, T, -1, -1)
        action_pad = torch.cat([action_pad_i, action_pad_t], dim=2)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.visual_encoder(padded_patches, lang_embed)

        # Get latent action for all future frames
        z_i = z[:, 1:, :self.num_i_codes]  # (B, T-1, n_i, E)
        z_t = z[:, 1:, self.num_i_codes: self.num_codes]  # (B, T-1, n_t, E)
        
        z_i = self.to_codebook_i(z_i)
        z_t = self.to_codebook_t(z_t)

        # Vector quantize separately for i and t
        z_i_flat = z_i.reshape(B * (T - 1), self.num_i_codes, self.latent_dim)
        z_t_flat = z_t.reshape(B * (T - 1), self.num_t_codes, self.latent_dim)
        
        z_q_i, z_orig_i, emb_i, indices_i = self.vq_i(z_i_flat)
        z_q_t, z_orig_t, emb_t, indices_t = self.vq_t(z_t_flat)
        
        z_q_i = z_q_i.reshape(B, T - 1, self.num_i_codes, self.latent_dim)
        z_q_t = z_q_t.reshape(B, T - 1, self.num_t_codes, self.latent_dim)
        
        # Concatenate the quantized results
        z_q = torch.cat([z_q_i, z_q_t], dim=2)
        z = torch.cat([z_orig_i, z_orig_t], dim=1)  # Concatenate for loss computation
        emb = torch.cat([emb_i, emb_t], dim=1)
        indices = torch.cat([indices_i, indices_t], dim=1)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }

    def forward(self, batch: Dict) -> Dict:
        # Visual encoding with supervised action prediction
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        outputs = self.visual_vq_encode(videos=batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode visual features
        video_recon = self.visual_decoder(x=video_action_patches)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]]

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        
        # Prepare visual features for action prediction (using t-codes part)
        visual_features = outputs["z"][:, self.num_i_codes: self.num_codes]  # [B, 1, num_t_codes, latent_dim]
        visual_features = visual_features.reshape(-1, self.latent_dim * self.num_t_codes)
        
        # Decode visual features to action space using action decoder
        predicted_action = self.action_decoder(visual_features)  # [B, max_action_dim]
        
        # Get ground truth action (first frame of the action sequence)
        # Assuming the action corresponds to the transition from frame 0 to frame 1
        target_action = batch["action"][:, 0]  # [B, Da]
        B, Da = target_action.shape
        
        # Pad target action if necessary
        if Da < self.max_action_dim:
            target_action = torch.cat([
                target_action, 
                torch.zeros(B, self.max_action_dim - Da, device=target_action.device)
            ], dim=1)
        
        # Compute supervised action prediction loss
        action_loss = F.mse_loss(predicted_action, target_action)
        outputs["loss_action"] = action_loss
        outputs["predicted_action"] = predicted_action
        outputs["target_action"] = target_action
        
        return outputs
    
    def action_vq_encode(self, action: Tensor, act_lens: Tensor = None) -> Dict:
        B, T, D = action.shape
        
        # Use MldVae encoder to get latent representation directly from action
        # Use act_lens if provided, otherwise assume all sequences have length T
        if act_lens is not None:
            lengths = act_lens.cpu().tolist()  # Convert tensor to list of integers
        else:
            lengths = [T] * B
        z, _ = self.action_vae.encode(action, lengths)  # z: [latent_size, B, latent_dim], dist not needed
        
        # Reshape z to match expected format [B, num_t_codes, latent_dim]
        z = z.permute(1, 0, 2)  # [B, latent_size, latent_dim]

        # Vector quantize
        z_q, z_orig, emb, indices = self.vq_t(z)
        return {
            "z_q": z_q,
            "z": z_orig,
            "emb": emb,
            "indices": indices,
        }
    
    def forward_action(self, batch: Dict) -> Dict:
        # Action VQ-VAE
        B, T, D = batch["action"].shape
        action = batch["action"]       # [B, T, D(7, 14)]
        if D < self.max_action_dim:
            action = torch.cat([action, torch.zeros(B, T, 14 - D).to(action.device)], dim=2)
        act_lens = batch["act_lens"]   # [B]
        
        # Use action_vq_encode with proper act_lens (timesteps handled internally by MldVae)
        outputs = self.action_vq_encode(action, act_lens)
        
        # Convert act_lens to proper lengths list for MldVae decoder
        lengths = act_lens.cpu().tolist()  # Convert tensor to list of integers
        
        # Use MldVae decoder to reconstruct action
        # Prepare z_q for decoder: [latent_size, B, latent_dim]
        z_q_decode = outputs['z_q'].permute(1, 0, 2)  # [latent_size, B, latent_dim]
        
        action_recon = self.action_vae.decode(z_q_decode, lengths)  # [B, T, latent_dim]
        
        outputs.update(
            {
                "recon": action_recon,
                "target": action
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
