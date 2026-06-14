from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
import torch.nn.functional as F
from accelerate import PartialState

OptimizerCallable = Callable[[Iterable], Optimizer]

from clap.modules import ContrastiveDINOLatentActionModel, LatentActionModel, DualBranchLatentActionModel
from clap.data_transform import AstribotPipeline, DeltaToAbsolute, ActionDenormalization
from clap.data_transform_droid import DROID_EXTERNAL_IMAGE_KEYS, DROID_SELECTED_EXTERNAL_KEY, DroidPipeline
from clap.unified_action import get_arm_mask, scatter_active_arm_actions
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)



class DINO_CLAP(LightningModule):
    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            action_space: str = "xyz_rpy",  # "xyz_rpy" or "xyz_r6d"
            clap_model_dim: int = 768,
            clap_latent_dim: int = 128,
            clap_action_vae_dim: int = 128,
            clap_num_t_codes: int = 4,
            clap_visual_t_codes: int = 8,
            clap_num_latents: int = 16,
            clap_patch_size: int = 16,
            clap_enc_blocks: int = 12,
            clap_dec_blocks: int = 12,
            clap_num_heads: int = 12,
            clap_dropout: float = 0.0,
            chunk_size: int = 16,
            action_layers: int = 9,
            use_residual_vq: bool = False,
            # DINO model configuration
            dino_model_type: str = "dinov3",
            dino_model_variant: str = "dinov3_vitb16",
            dino_model_path: str = "facebookresearch/dinov3",
            dino_weights_path: str = "~/models/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            mse_alpha: float = 1.0,
            siglip_alpha: float = 1.0,
            reg_alpha: float = 0.0,
            vq_beta: float = 0.25,
            filter_fc: float = 8,
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
            # Learning rate scheduler configuration
            warmup_steps: int = 200,
            lr_scheduler_type: str = 'constant',  # 'constant', 'cosine', 'linear'
            max_training_steps: int = 50000,  # Total training steps for cosine/linear decay
            min_lr_ratio: float = 0.1,  # Minimum learning rate ratio for cosine decay
            codebook_lr_scale: float = 1.0,  # Learning rate scale for codebook parameters
    ) -> None:
        super(DINO_CLAP, self).__init__()
        assert stage in ['stage-1', 'stage-2', 'dual']
        # stage-1: Action VQ-VAE
        # stage-2: Visual CLAP
        self.stage = stage
        self._strict_loading = False
        
        # Set action space and dimensions
        assert action_space == "xyz_rpy", f"Only xyz_rpy is supported, got {action_space}"
        self.action_space = action_space
        # OpenCLAP only ships xyz_rpy (14-D dual-arm) — drop the r6d branch.
        self.action_dim_per_arm = 7  # xyz(3) + rpy(3) + gripper(1)
        self.astribot_pipeline = AstribotPipeline(fc=filter_fc)
        self.droid_pipeline = DroidPipeline()

        module_dict = {
            'stage-1': LatentActionModel,
            'stage-2': ContrastiveDINOLatentActionModel,
            'dual': DualBranchLatentActionModel,
        }
        _CLAP = module_dict[stage]

        self.clap = _CLAP(
            in_dim=image_channels,
            model_dim=clap_model_dim,
            latent_dim=clap_latent_dim,
            max_action_dim=self.action_dim_per_arm,
            action_vae_dim=clap_action_vae_dim,
            chunk_size=chunk_size,
            num_latents=clap_num_latents,
            num_t_codes=clap_num_t_codes,
            visual_t_codes=clap_visual_t_codes,
            patch_size=clap_patch_size,
            enc_blocks=clap_enc_blocks,
            dec_blocks=clap_dec_blocks,
            num_heads=clap_num_heads,
            action_layers=action_layers,
            dropout=clap_dropout,
            use_residual_vq=use_residual_vq,
            dino_model_type=dino_model_type,
            dino_model_variant=dino_model_variant,
            dino_model_path=dino_model_path,
            dino_weights_path=dino_weights_path,
        )
        self.delta_to_absolute = DeltaToAbsolute(action_space=action_space)
        self.action_denormalization = ActionDenormalization(robot_id=0, action_space=action_space)
        
        if stage_one_ckpt and path.exists(stage_one_ckpt):
            print(f"Loading stage-1 CLAP checkpoint from {stage_one_ckpt}")
            clap_ckpt = torch.load(stage_one_ckpt, map_location='cpu')['state_dict']
            clap_ckpt = {k.replace("clap.", ""): v for k, v in clap_ckpt.items()}
            self.clap.load_state_dict(clap_ckpt, strict=False)
        if hasattr(self.clap, "reset_ema_encoder"):
            self.clap.reset_ema_encoder()


        self.clap_num_latents = clap_num_latents
        self.mse_alpha = mse_alpha
        self.siglip_alpha = siglip_alpha
        self.reg_alpha = reg_alpha
        self.vq_beta = vq_beta
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair
        self.warmup_steps = warmup_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.max_training_steps = max_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.codebook_lr_scale = codebook_lr_scale

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        
        self.test_results = []
        self.test_meta_data = {}

    def _preprocess_batch(self, batch: Dict) -> Dict:
        if "action" in batch and "state" in batch and "arm_mask" in batch:
            return batch
        robot_type = batch.get("robot_type")
        if isinstance(robot_type, (list, tuple)) and any(rt == "droid" for rt in robot_type):
            if self.droid_pipeline is None:
                raise ValueError("DROID preprocessing is only supported for xyz_rpy action space")
            if DROID_SELECTED_EXTERNAL_KEY not in batch:
                for key in DROID_EXTERNAL_IMAGE_KEYS:
                    if key in batch:
                        batch[DROID_SELECTED_EXTERNAL_KEY] = batch[key]
                        break
            return self.droid_pipeline(batch)
        if robot_type == "droid":
            if self.droid_pipeline is None:
                raise ValueError("DROID preprocessing is only supported for xyz_rpy action space")
            if DROID_SELECTED_EXTERNAL_KEY not in batch:
                for key in DROID_EXTERNAL_IMAGE_KEYS:
                    if key in batch:
                        batch[DROID_SELECTED_EXTERNAL_KEY] = batch[key]
                        break
            return self.droid_pipeline(batch)
        return self.astribot_pipeline(batch)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['observation.head', 'subtask', 'action', 'dataset_name']

        outputs = self.clap(batch)
        gt_future_frames = outputs["target"]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = self.mse_alpha * mse_loss + q_loss + self.vq_beta * commit_loss

        # Compute code usage
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.clap_num_latents, dtype=torch.long, device=outputs["indices"].device)
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
        )
        if "loss_var_len" in outputs:
            loss_var_len = outputs["loss_var_len"]
            loss_logs = loss_logs + (("loss_var_len", loss_var_len),)
            loss = loss + self.mse_alpha * loss_var_len * 0.1
        
        if self.stage == 'stage-2':
            loss_siglip = outputs["loss_siglip"]
            loss_logs = loss_logs + (("loss_siglip", loss_siglip),)
            reg_loss = outputs["reg_loss"]
            loss_logs = loss_logs + (("reg_loss", reg_loss),)
            loss = loss + self.siglip_alpha * loss_siglip
            loss = loss + self.reg_alpha * reg_loss
        elif self.stage == 'dual':
            loss_action = outputs["loss_action"]
            loss_logs = loss_logs + (("loss_action", loss_action),)
            loss = loss + self.mse_alpha * loss_action

        return outputs, loss, loss_logs


    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        if (self.global_step + 1) % 2000 == 0:
            self.clap.random_restart()
        
        # Compute the training loss
        batch = self._preprocess_batch(batch)
        outputs, loss, aux_losses = self.shared_step(batch)

        log_dict = {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}}
        
        # Log learning rates for different parameter groups
        optimizer = self.optimizers()
        for idx, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{idx}')
            log_dict[f'lr_{group_name}'] = param_group['lr']
        
        # Log the training loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self.clap, "update_ema_encoder"):
            self.clap.update_ema_encoder()
    
    def on_before_optimizer_step(self, optimizer):
        """
        Log gradient norms for each parameter group.
        Uses PyTorch's native gradient norm computation.
        """
        # Compute and log gradient norms for each parameter group
        for idx, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{idx}')
            
            # Collect parameters with gradients
            grads = []
            for p in param_group['params']:
                if p.grad is not None:
                    grads.append(p.grad.detach())
            
            if len(grads) > 0:
                # Compute L2 norm of all gradients in this group
                # Stack all gradients and compute norm
                total_norm = torch.norm(
                    torch.stack([torch.norm(g, 2.0) for g in grads]),
                    2.0
                )
                
                self.log(
                    f'grad_norm/{group_name}',
                    total_norm,
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True
                )

    @torch.no_grad()
    def val_step(self, batch: Dict, batch_idx: int) -> Tensor:
        self.test_step(batch, batch_idx)
    
    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        if batch["frame_index"][0].item() % self.clap.chunk_size != 0:
            return

        self.test_meta_data = {
            "dataset_name": batch["dataset_name"][0],
            "episode": batch["episode_index"][0].item(),
        }

        batch = self._preprocess_batch(batch)
    
        if self.stage == 'stage-1':
            outputs_action = self.clap(batch)
        elif self.stage == 'stage-2':
            outputs_action = self.clap.forward_action(batch)
        action_org = batch["action"].float()  # [B, T, 2*action_dim_per_arm]
        action_vae_output = outputs_action["recon"].float()  # [2*B, T, action_dim_per_arm] - left and right concatenated in batch dim
        B = action_org.shape[0]
        T = action_org.shape[1]
        arm_mask = get_arm_mask(
            batch,
            B,
            default_layout="dual",
            device=action_org.device,
            dtype=torch.bool,
        )
        
        # Convert delta actions to absolute actions using the first state
        # Delta actions are computed relative to the first state in data_transform.py
        state_first = batch["state"][:, :1].float()  # [B, 1, 2*action_dim_per_arm]
        
        # Get robot_id from batch if available, otherwise use default (0)
        robot_id = batch.get("robot_id", torch.tensor([0]))[0].item() if "robot_id" in batch else 0
        
        # Scatter active arm reconstructions back into canonical [left, right] slots.
        action_vae = scatter_active_arm_actions(action_vae_output, arm_mask, self.action_dim_per_arm)
        
        # Denormalize actions and state back to original scale
        action_org = self.action_denormalization.denormalize_action(action_org, robot_id)
        action_vae = self.action_denormalization.denormalize_action(action_vae, robot_id)
        state_first = self.action_denormalization.denormalize_state(state_first, robot_id)
        
        # Split into left and right after denormalization
        action_org_left = action_org[..., :self.action_dim_per_arm]
        action_org_right = action_org[..., self.action_dim_per_arm:2*self.action_dim_per_arm]
        action_vae_left = action_vae[..., :self.action_dim_per_arm]
        action_vae_right = action_vae[..., self.action_dim_per_arm:2*self.action_dim_per_arm]
        state_left_first = state_first[..., :self.action_dim_per_arm]
        state_right_first = state_first[..., self.action_dim_per_arm:2*self.action_dim_per_arm]
        
        # Convert delta to absolute
        # Expand state_first to match action dimensions [B, T, 7]
        state_left_expanded = state_left_first.expand(-1, T, -1)
        state_right_expanded = state_right_first.expand(-1, T, -1)
        
        action_org_left = self.delta_to_absolute.convert(action_org_left, state_left_expanded)
        action_org_right = self.delta_to_absolute.convert(action_org_right, state_right_expanded)
        action_vae_left = self.delta_to_absolute.convert(action_vae_left, state_left_expanded)
        action_vae_right = self.delta_to_absolute.convert(action_vae_right, state_right_expanded)

        left_active = arm_mask[:, 0].view(B, 1, 1).to(action_org_left.dtype)
        right_active = arm_mask[:, 1].view(B, 1, 1).to(action_org_right.dtype)
        action_org_left = action_org_left * left_active
        action_org_right = action_org_right * right_active
        action_vae_left = action_vae_left * left_active
        action_vae_right = action_vae_right * right_active
        
        result_dict = {
            "action_org_left": action_org_left,
            "action_org_right": action_org_right,
            "action_vae_left": action_vae_left,
            "action_vae_right": action_vae_right,
        }
        if self.stage != 'stage-1':
            result_dict["image"] = batch["observation.head"][:, :1]
        
        # Only compute visual branch for stage-2 and dual
        if self.stage in ['stage-2', 'dual']:
            visual_outputs = self.clap.visual_vq_encode(videos=batch["observation.head"])
            visual_z_q = visual_outputs["z_q"]  # [B, 1, num_codes, latent_dim]
            visual_action_codes = visual_z_q[:, 0, self.clap.num_i_codes: self.clap.num_codes]
            num_t_codes = self.clap.visual_t_codes
            
            # Split into left and right arm features
            visual_z_q_left = visual_action_codes[:, :num_t_codes]  # [B, num_t_codes, latent_dim]
            visual_z_q_right = visual_action_codes[:, num_t_codes:]  # [B, num_t_codes, latent_dim]
            
            # Decode visual features to actions using action_vae decoder
            # Permute to [num_t_codes, B, latent_dim] format required by MldVae.decode
            z_q_left_decode = visual_z_q_left.permute(1, 0, 2)
            z_q_right_decode = visual_z_q_right.permute(1, 0, 2)
            lengths = [T] * B
            action_visual_left = self.clap.action_vae.decode(z_q_left_decode, lengths).float()  # [B, T, action_dim_per_arm]
            action_visual_right = self.clap.action_vae.decode(z_q_right_decode, lengths).float()  # [B, T, action_dim_per_arm]
            
            # Concatenate left and right arms to form dual-arm action [B, T, 2*action_dim_per_arm]
            action_visual = torch.cat([action_visual_left, action_visual_right], dim=-1)
            
            # Denormalize visual actions (requires 2*action_dim_per_arm dual arm)
            action_visual = self.action_denormalization.denormalize_action(action_visual, robot_id)
            
            # Split back into left and right after denormalization
            action_visual_left = action_visual[..., :self.action_dim_per_arm]
            action_visual_right = action_visual[..., self.action_dim_per_arm:2*self.action_dim_per_arm]
            
            # Convert visual delta actions to absolute actions
            action_visual_left = self.delta_to_absolute.convert(action_visual_left, state_left_expanded)
            action_visual_right = self.delta_to_absolute.convert(action_visual_right, state_right_expanded)
            action_visual_left = action_visual_left * left_active
            action_visual_right = action_visual_right * right_active
            
            result_dict["action_visual_left"] = action_visual_left
            result_dict["action_visual_right"] = action_visual_right

        self.test_results.append(result_dict)

    def on_train_epoch_end(self):
        if self.stage in ['stage-2', 'dual']:
            self.clap.vq_i.random_restart()
            self.clap.vq_i.reset_usage()
        self.clap.vq_t.random_restart()
        self.clap.vq_t.reset_usage()

    def on_validation_epoch_start(self):
        """Clear test results at the start of testing"""
        self.test_results = []
        self.test_meta_data = {}
    
    def on_validation_epoch_end(self):
        self.on_test_epoch_end()
    
    def on_test_epoch_start(self):
        """Clear test results at the start of testing"""
        self.test_results = []
        self.test_meta_data = {}

    def on_test_epoch_end(self):
        test_results = self.test_results
        
        def fuse_batch_results(results, key):
            res = [result[key] for result in results if key in result]
            if len(res) == 0:
                return None
            res = torch.cat(res, dim=0)
            res = torch.flatten(res, start_dim=0, end_dim=1)
            return res.cpu().numpy()
        
        if self.stage != 'stage-1':
            image = fuse_batch_results(test_results, "image")
        else:
            image = None
        
        # Fuse actions (already converted to absolute values in test_step)
        action_org_left = fuse_batch_results(test_results, "action_org_left")
        action_org_right = fuse_batch_results(test_results, "action_org_right")
        action_vae_left = fuse_batch_results(test_results, "action_vae_left")
        action_vae_right = fuse_batch_results(test_results, "action_vae_right")
        action_visual_left = fuse_batch_results(test_results, "action_visual_left")
        action_visual_right = fuse_batch_results(test_results, "action_visual_right")
        
        # Visualize the results
        self.visualize_test_results(
            image, 
            action_org_left, action_org_right,
            action_vae_left, action_vae_right,
            action_visual_left, action_visual_right
        )
    
    def visualize_test_results(self, image, action_org_left, action_org_right,
                               action_vae_left, action_vae_right,
                               action_visual_left, action_visual_right):
        """
        Visualize test results with separate plots for each DOF:
        - 7 rows x 2 columns for 7 DOFs of left and right arms
        - Each subplot shows GT, VAE recon, and Visual prediction (if available)
        - Supports image=None for stage-1 where images are not available
        """
        # Check if visual predictions are available (stage-2 or dual)
        has_visual = (action_visual_left is not None) and (action_visual_right is not None)
        
        # Check if images are available (stage-2 or dual)
        has_images = image is not None
        
        # Create figure with (action_dim_per_arm+1) rows x 2 columns (first row for images + MSE, then action_dim_per_arm rows for DOFs)
        num_dofs = self.action_dim_per_arm
        fig = plt.figure(figsize=(20, 4 * (num_dofs + 1)))
        gs = fig.add_gridspec(num_dofs + 1, 2, hspace=0.4, wspace=0.3, height_ratios=[1.2] + [1]*num_dofs)
        
        # Row 0, Column 0: Sample images (if available)
        ax_img = fig.add_subplot(gs[0, 0])
        if has_images:
            num_samples = min(8, len(image))
            img_grid = np.zeros((2, 4, *image.shape[1:]))  # [2, 4, C, H, W]
            for i in range(num_samples):
                row, col = i // 4, i % 4
                img_grid[row, col] = image[i * len(image) // num_samples]
            
            # Rearrange for display: [2, 4, C, H, W] -> [2*H, 4*W, C]
            img_grid = np.transpose(img_grid, (0, 3, 1, 4, 2))  # [2, H, 4, W, C]
            img_grid = img_grid.reshape(img_grid.shape[0] * img_grid.shape[1], 
                                         img_grid.shape[2] * img_grid.shape[3], 
                                         img_grid.shape[4])  # [2*H, 4*W, C]
            img_grid = np.clip(img_grid, 0, 1)
            ax_img.imshow(img_grid)
            ax_img.set_title('Sample Images', fontsize=14, fontweight='bold')
        else:
            # Display placeholder text when images are not available
            ax_img.text(0.5, 0.5, 'No Images Available\n(Stage-1)', 
                       ha='center', va='center', fontsize=16, fontweight='bold',
                       transform=ax_img.transAxes)
            ax_img.set_title('Sample Images', fontsize=14, fontweight='bold')
        ax_img.axis('off')
        
        # Row 0, Column 1: MSE comparison
        ax_mse = fig.add_subplot(gs[0, 1])
        self._plot_mse_comparison(ax_mse, 
                                  action_org_left, action_org_right,
                                  action_vae_left, action_vae_right,
                                  action_visual_left, action_visual_right,
                                  has_visual)
        
        # DOF names based on action_space
        if self.action_space == "xyz_rpy":
            dof_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
        num_samples = min(500, len(action_org_left))
        timesteps = np.arange(num_samples)
        
        for dof_idx in range(num_dofs):
            # Left arm - Column 0
            ax_left = fig.add_subplot(gs[dof_idx + 1, 0])
            visual_left_dof = action_visual_left[:num_samples, dof_idx] if has_visual else None
            self._plot_single_dof(ax_left, timesteps,
                                 action_org_left[:num_samples, dof_idx],
                                 action_vae_left[:num_samples, dof_idx],
                                 visual_left_dof,
                                 f'Left Arm - {dof_names[dof_idx]}')
            
            # Right arm - Column 1
            ax_right = fig.add_subplot(gs[dof_idx + 1, 1])
            visual_right_dof = action_visual_right[:num_samples, dof_idx] if has_visual else None
            self._plot_single_dof(ax_right, timesteps,
                                 action_org_right[:num_samples, dof_idx],
                                 action_vae_right[:num_samples, dof_idx],
                                 visual_right_dof,
                                 f'Right Arm - {dof_names[dof_idx]}')
        
        # Save figure
        log_path = f"./logs/{self.task_name}"
        save_path = f"{log_path}"\
            f"/test_results_step_{self.global_step}"\
            f"_dataset_{self.test_meta_data['dataset_name']}"\
            f"_episode_{self.test_meta_data['episode']}.jpg"
            
        makedirs(log_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Log to wandb if main process
        if self.distributed_state.is_main_process:
            wandb.log({"test_visualization": wandb.Image(save_path)})
        
        plt.close(fig)
        logging.info(f"Test visualization saved to {save_path}")
    
    def _plot_single_dof(self, ax, timesteps, gt, vae, visual, title):
        """Plot action curves for a single DOF"""
        # Plot ground truth and VAE reconstruction
        ax.plot(timesteps, gt, label='Ground Truth', 
               linewidth=2.0, alpha=0.9, color='#2E86AB')
        ax.plot(timesteps, vae, label='VAE Recon', 
               linewidth=2.0, alpha=0.8, linestyle='--', color='#A23B72')
        
        # Compute MSE for VAE
        mse_vae = np.mean((gt - vae) ** 2)
        textstr = f'MSE (VAE): {mse_vae:.4f}'
        
        # Plot visual prediction if available
        if visual is not None:
            ax.plot(timesteps, visual, label='Visual Pred', 
                   linewidth=2.0, alpha=0.8, linestyle=':', color='#F18F01')
            mse_visual = np.mean((gt - visual) ** 2)
            textstr += f'\nMSE (Visual): {mse_visual:.4f}'
        
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Display MSE in the plot
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_mse_comparison(self, ax, action_org_left, action_org_right,
                            action_vae_left, action_vae_right,
                            action_visual_left, action_visual_right,
                            has_visual):
        """Plot MSE comparison for each DOF"""
        # Compute MSE for each DOF
        mse_vae_left_per_dof = np.mean((action_org_left - action_vae_left) ** 2, axis=0)  # [action_dim_per_arm]
        mse_vae_right_per_dof = np.mean((action_org_right - action_vae_right) ** 2, axis=0)  # [action_dim_per_arm]
        
        # DOF names based on action_space
        if self.action_space == "xyz_rpy":
            dof_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Grip']
        x = np.arange(len(dof_names))
        
        if has_visual:
            # Stage-2 or dual: show VAE and Visual predictions
            mse_visual_left_per_dof = np.mean((action_org_left - action_visual_left) ** 2, axis=0)  # [action_dim_per_arm]
            mse_visual_right_per_dof = np.mean((action_org_right - action_visual_right) ** 2, axis=0)  # [action_dim_per_arm]
            
            width = 0.2
            # Plot bars for left and right, VAE and Visual
            bars1 = ax.bar(x - 1.5*width, mse_vae_left_per_dof, width, 
                           label='Left VAE', alpha=0.8, color='#2E86AB')
            bars2 = ax.bar(x - 0.5*width, mse_visual_left_per_dof, width,
                           label='Left Visual', alpha=0.8, color='#A23B72')
            bars3 = ax.bar(x + 0.5*width, mse_vae_right_per_dof, width,
                           label='Right VAE', alpha=0.8, color='#F18F01')
            bars4 = ax.bar(x + 1.5*width, mse_visual_right_per_dof, width,
                           label='Right Visual', alpha=0.8, color='#C73E1D')
        else:
            # Stage-1: only show VAE predictions
            width = 0.35
            bars1 = ax.bar(x - width/2, mse_vae_left_per_dof, width, 
                           label='Left VAE', alpha=0.8, color='#2E86AB')
            bars2 = ax.bar(x + width/2, mse_vae_right_per_dof, width,
                           label='Right VAE', alpha=0.8, color='#F18F01')
        
        ax.set_ylabel('Mean Squared Error', fontsize=11)
        ax.set_title('MSE Comparison per DOF (Lower is Better)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dof_names, fontsize=10)
        ax.legend(fontsize=9, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')  # Use log scale for better visualization
        
        # Log overall metrics to wandb
        if self.distributed_state.is_main_process:
            mse_vae_left_overall = np.mean(mse_vae_left_per_dof)
            mse_vae_right_overall = np.mean(mse_vae_right_per_dof)
            
            log_dict = {
                "test/mse_vae_left_overall": mse_vae_left_overall,
                "test/mse_vae_right_overall": mse_vae_right_overall,
                # Log per-DOF metrics for VAE
                **{f"test/mse_vae_left_{dof_names[i]}": mse_vae_left_per_dof[i] for i in range(self.action_dim_per_arm)},
                **{f"test/mse_vae_right_{dof_names[i]}": mse_vae_right_per_dof[i] for i in range(self.action_dim_per_arm)},
            }
            
            if has_visual:
                mse_visual_left_overall = np.mean(mse_visual_left_per_dof)
                mse_visual_right_overall = np.mean(mse_visual_right_per_dof)
                log_dict.update({
                    "test/mse_visual_left_overall": mse_visual_left_overall,
                    "test/mse_visual_right_overall": mse_visual_right_overall,
                    # Log per-DOF metrics for Visual
                    **{f"test/mse_visual_left_{dof_names[i]}": mse_visual_left_per_dof[i] for i in range(self.action_dim_per_arm)},
                    **{f"test/mse_visual_right_{dof_names[i]}": mse_visual_right_per_dof[i] for i in range(self.action_dim_per_arm)},
                })
            
            wandb.log(log_dict)

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def _get_lr_scheduler_lambda(self):
        """
        Create learning rate scheduler lambda function based on scheduler type.
        Supports: constant, cosine, linear with warmup.
        """
        import math
        
        def constant_with_warmup(current_step: int):
            """Linear warmup then constant."""
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0
        
        def cosine_with_warmup(current_step: int):
            """Linear warmup then cosine annealing."""
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            progress = (current_step - self.warmup_steps) / max(1, self.max_training_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
        
        def linear_with_warmup(current_step: int):
            """Linear warmup then linear decay."""
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            
            progress = (current_step - self.warmup_steps) / max(1, self.max_training_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (1.0 - progress)
        
        # Select scheduler based on type
        if self.lr_scheduler_type == 'constant':
            return constant_with_warmup
        elif self.lr_scheduler_type == 'cosine':
            return cosine_with_warmup
        elif self.lr_scheduler_type == 'linear':
            return linear_with_warmup
        else:
            logging.warning(f"Unknown lr_scheduler_type: {self.lr_scheduler_type}, using constant schedule")
            return constant_with_warmup
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Features:
            - Separate weight decay for codebook (0.0) and other parameters
            - Separate learning rate scale for codebook parameters
            - Warmup + constant/cosine/linear learning rate schedule
        """
        # Separate parameters: codebook without weight decay, encoder/decoder with weight decay
        codebook_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'codebook' in name:
                codebook_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different weight decay
        param_groups = [
            {'params': other_params, 'name': 'encoder_decoder'},  # encoder/decoder: inherit weight_decay from config
            {'params': codebook_params, 'weight_decay': 0.0, 'name': 'codebook'}  # codebook: no weight decay
        ]
        
        optim = self.optimizer(param_groups)
        
        # Create learning rate scheduler with separate scale for codebook
        from torch.optim.lr_scheduler import LambdaLR
        
        base_lr_lambda = self._get_lr_scheduler_lambda()
        
        # Create separate lr_lambda for each parameter group
        # encoder_decoder uses base schedule, codebook uses scaled schedule
        lr_lambdas = [
            base_lr_lambda,  # encoder_decoder
            lambda step: base_lr_lambda(step) * self.codebook_lr_scale  # codebook with scale
        ]
        
        scheduler = LambdaLR(optim, lr_lambda=lr_lambdas)
        
        logging.info(f"Learning rate scheduler: {self.lr_scheduler_type}")
        logging.info(f"  - Codebook LR scale: {self.codebook_lr_scale}")
        if self.lr_scheduler_type in ['cosine', 'linear']:
            logging.info(f"  - Warmup steps: {self.warmup_steps}")
            logging.info(f"  - Max training steps: {self.max_training_steps}")
            logging.info(f"  - Min LR ratio: {self.min_lr_ratio}")
        
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,
            },
        }
