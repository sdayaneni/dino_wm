import torch
from torch import nn
import pytorch_lightning as L

class EBT(L.LightningModule):
    def __init__(
        self, 
        world_model, 
        num_mcmc_steps,
        action_dim, 
        mcmc_step_size, 
        mcmc_step_size_learnable,
        learning = True, 
        image_dims, 
        patch_size, 
        embedding_dim,
        num_transformer_blocks,
        multiheaded_attention_heads,
        ffn_dim_multiplier
    ):
        super().__init__()
        
        # READ IN HYDRA HPARAMS HERE
        self.num_mcmc_steps = num_mcmc_steps
        self.action_dim = action_dim
        self.mcmc_step_size_learnable = mcmc_step_size_learnable
        self.alpha = nn.Parameter(torch.tensor(float(mcmc_step_size)), requires_grad=self.mcmc_step_size_learnable)
        self.learning = learning
        self.image_dims = image_dims
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.multiheaded_attention_heads = multiheaded_attention_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        
        # Freeze world model
        self.world_model = world_model
        for param in self.world_model.parameters():
            param.requires_grad = False
        self.world_model.eval()
        
        self.transformer = self.setup_bidirectional_ebt()
        self.loss_fn = nn.MSELoss()
        
        self.action_dim = self.hparams.get('action_dim', None)
        if self.action_dim is None:
            raise ValueError("action_dim must be specified in hparams")
        
    def get_energy(self, predicted_states, goal_states):
        goal_visual = goal_states['visual']
        goal_proprio = goal_states['proprio']    

        # Get final timestep
        pred_visual = predicted_states['visual'][:, -1:, :, :]  # (B, 1, num_patches, emb_dim)
        pred_proprio = predicted_states['proprio'][:, -1:, :]  # (B, 1, proprio_dim)
        
        pred_visual = pred_visual.squeeze(1)  # (B, num_patches, emb_dim)
        goal_visual = goal_visual.squeeze(1)  # (B, num_patches, emb_dim)
        
        all_embeddings = torch.cat([pred_visual, goal_visual], dim=1)  # (B, 2*num_patches, emb_dim)

        B, num_patches, D = all_embeddings.shape

        spatial_size = int(total_patches ** 0.5)
        if spatial_size * spatial_size < total_patches:
            spatial_size += 1
            pad_size = spatial_size * spatial_size - total_patches
            padding = torch.zeros(B, pad_size, D, device=all_embeddings.device, dtype=all_embeddings.dtype)
            all_embeddings = torch.cat([all_embeddings, padding], dim=1)
        
        spatial_input = all_embeddings.transpose(1, 2).reshape(B, D, spatial_size, spatial_size)

        zero_condition = torch.zeros(B, self.embedding_dim, device=all_embeddings.device)
        
        energy_preds = self.transformer(spatial_input, zero_condition).squeeze()
        energy_preds = energy_preds.mean(dim=[1]).reshape(-1) # B

        # TODO: maybe replicate for proprio??
            
        return energy_preds

    def setup_bidirectional_ebt(self):
        # TODO: modify for DINO
        from models.bi_ebt_adaln import EBT
        assert self.image_dims[0] == self.image_dims[1], "need to use square image with current implementation"

        if hparams.image_task == "denoising":
            # For denoising task, use raw image dimensions (no VAE)
            input_size = hparams.image_dims[0]
            in_channels = 3  # RGB channels for raw images
        else:
            # For other tasks using VAE
            assert self.image_dims[0] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
            input_size = self.image_dims[0] // 8
            in_channels = 4
    
        ebt = EBT(
            input_size=input_size,
            patch_size=self.patch_size,
            in_channels=in_channels,
            hidden_size=self.embedding_dim,
            depth=self.num_transformer_blocks,
            num_heads=self.multiheaded_attention_heads,
            mlp_ratio=self.ffn_dim_multiplier
        )
        
        return ebt
    
    def forward(self, batch):
        obs, act, state = batch
        
        batch_size = obs['visual'].shape[0]
        num_frames = obs['visual'].shape[1]
        num_hist = self.world_model.num_hist # Historical actions (given to rollout)

        init_states = {
            'visual': obs['visual'][:, :num_hist, ...],
            'proprio': obs['proprio'][:, :num_hist, ...]
        }
        
        goal_states = {
            'visual': obs['visual'][:, -1:, ...],
            'proprio': obs['proprio'][:, -1:, ...]
        }
        
        gt_actions = act[:, num_hist-1:num_frames-1, :]  # TODO: Adjust indexing if needed
        
        action_dim = gt_actions.shape[2]
        
        # Random
        actions = torch.randn(batch_size, num_frames, action_dim, device=init_states['visual'].device, requires_grad=True)
        
        alpha = torch.clamp(self.alpha, min=0.0001)
        energy_history = []
        
        # MCMC loop
        for i in range(self.num_mcmc_steps):
            num_init_obs = init_states['visual'].shape[1]
            
            with torch.set_grad_enabled(True):
                predicted_states, _ = self.world_model.rollout(obs_0=init_states, act=actions)
            
            energy = self.get_energy(predicted_states, goal_states)
            energy_sum = energy.sum()
            energy_history.append(energy.detach().mean().item())
            
            action_grad = torch.autograd.grad(energy_sum, actions, create_graph=self.learning)[0]
            
            if torch.isnan(action_grad).any() or torch.isinf(action_grad).any():
                raise ValueError("NaN or Inf gradients detected during MCMC.")
            
            with torch.no_grad():
                actions = actions - alpha * action_grad
                actions.requires_grad_(True)
        
        return actions[:, num_hist:, :], energy_history
    
    def training_step(self, batch):
        """
        Training step: optimize actions via MCMC, then compute loss vs GT actions.
        """
        actions, energy_history = self(batch)
        obs, act, state = batch 
        num_hist = self.world_model.num_hist
        num_frames = obs['visual'].shape[1]
        gt_actions = act[:, num_hist-1:num_frames-1, :]
        loss = self.loss_fn(actions, gt_actions)
        
        return loss
    
    def setup_optim(self):
        # Add EBM params
        params = list(self.transformer.parameters())
        if self.alpha.requires_grad:
            params.append(self.alpha)
        
        # Remove DINO params
        world_model_param_ids = {id(p) for p in self.world_model.parameters()}
        params = [p for p in params if id(p) not in world_model_param_ids]
        
        optimizer = torch.optim.Adam(params, lr=self.hparams.get('lr', 1e-4))
        
        return optimizer