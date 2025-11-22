import torch
from torch import nn
import random
import pytorch_lightning as L

class EBMPC(L.LightningModule):
    def __init__(
        self, 
        world_model, 
        num_mcmc_steps,
        action_dim, 
        mcmc_step_size, 
        mcmc_step_size_learnable,
        image_dims, 
        embedding_dim,
        num_transformer_blocks,
        multiheaded_attention_heads,
        ffn_dim_multiplier,
        lr,
        learning = True
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['world_model'])
        
        self.num_mcmc_steps = num_mcmc_steps
        self.action_dim = action_dim
        self.mcmc_step_size_learnable = mcmc_step_size_learnable
        self.alpha = nn.Parameter(torch.tensor(float(mcmc_step_size)), requires_grad=self.mcmc_step_size_learnable)
        self.learning = learning
        self.image_dims = image_dims
        self.num_transformer_blocks = num_transformer_blocks
        self.multiheaded_attention_heads = multiheaded_attention_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.lr = lr
        
        # Freeze world model
        self.world_model = world_model
        
        self.embedding_dim = self.world_model.encoder.emb_dim
        self.patch_size = self.world_model.encoder.patch_size
        
        self.transformer = self.setup_bidirectional_ebt()
        self.loss_fn = nn.MSELoss()
        
        if self.action_dim is None:
            raise ValueError("action_dim must be specified")
    
    def get_energy(self, predicted_states, goal_visual):
        pred_visual = predicted_states['visual'][:, -1:, :, :]  # (B, 1, T, D)
        
        pred_visual = pred_visual.squeeze(1)  # (B, T, D)
        
        all_embeddings = torch.cat([pred_visual, goal_visual], dim=1)  # (B, 2T, D)

        B, num_patches, D = all_embeddings.shape

        # Create custom positional encoding tensor (only add for goal)
        goal_pos_encoding = self.transformer.pos_embed[:, :num_patches//2, :]
        pred_pos_encoding = torch.zeros_like(goal_pos_encoding)
        pos_embed = torch.cat([pred_pos_encoding, goal_pos_encoding], dim=1)

        zero_condition = torch.zeros(B, self.embedding_dim, device=all_embeddings.device)
        
        energy_preds = self.transformer(x=all_embeddings, y=zero_condition, pos_embed=pos_embed)
        energy_preds = energy_preds.squeeze(-1)  # (B,)

        # TODO: maybe replicate for proprio??
            
        return energy_preds

    def setup_bidirectional_ebt(self):
        from models.bi_ebt_adaln import EBT
        assert self.image_dims[0] == self.image_dims[1], "need to use square image with current implementation"
        
        image_size = self.image_dims[0]
        patches_per_dim = image_size // self.patch_size
        patches_per_image = patches_per_dim ** 2
    
        ebt = EBT(
            input_size=image_size,
            patch_size=self.patch_size,
            in_channels=3,
            hidden_size=self.embedding_dim,
            depth=self.num_transformer_blocks,
            num_heads=self.multiheaded_attention_heads,
            mlp_ratio=self.ffn_dim_multiplier
        )

        ebt.pos_embed = nn.Parameter(torch.zeros(1, patches_per_image, self.embedding_dim), requires_grad=True)
        nn.init.xavier_uniform_(ebt.pos_embed.data)
        
        return ebt
    
    def forward(self, batch, random_end_point):
        obs, act, state = batch
        num_hist = self.world_model.num_hist

        init_states = {
            'visual': obs['visual'][:, :num_hist, ...],
            'proprio': obs['proprio'][:, :num_hist, ...]
        }
        
        goal_states = {
            'visual': obs['visual'][:, random_end_point:random_end_point+1, ...],   # (B,1,3,H,W)
            'proprio': obs['proprio'][:, random_end_point:random_end_point+1, ...]
        }

        goal_z = self.world_model.encode_obs(goal_states)  # {'visual': (B,num_frames,T,D)}
        goal_visual = goal_z['visual'][:, -1, :, :].squeeze(1)   # (B, T, D)
        
        gt_actions = act[:, :random_end_point, :]
        actions = torch.randn_like(gt_actions, device=init_states['visual'].device, requires_grad=True)
        
        alpha = torch.clamp(self.alpha, min=0.0001)
        energy_history = []
        action_grad_norms = []
        
        # MCMC loop
        for i in range(self.num_mcmc_steps):
            
            with torch.set_grad_enabled(True):
                rollout_obses, rollout_states = self.world_model.rollout(obs_0=init_states, act=actions)

                energy = self.get_energy(rollout_obses, goal_visual)
                energy_sum = energy.sum()
                energy_history.append(energy.detach().mean().item())
                
                action_grad = torch.autograd.grad(energy_sum, actions, create_graph=self.learning)[0]

            # Track gradient norms
            grad_norm = action_grad.norm().item()
            action_grad_norms.append(grad_norm)
            
            if torch.isnan(action_grad).any() or torch.isinf(action_grad).any():
                raise ValueError("NaN or Inf gradients detected during MCMC.")
            
            actions = (actions - alpha * action_grad).requires_grad_(True)
        
        return actions, energy_history, action_grad_norms
    
    def training_step(self, batch, batch_idx):
        """
        Training step: optimize actions via MCMC, then compute loss vs GT actions.
        """
        obs, act, state = batch

        num_hist = self.world_model.num_hist
        
        num_frames = obs['visual'].shape[1]
        num_proprio_frames = obs['proprio'].shape[1]

        max_valid = min(num_frames, num_proprio_frames)

        random_end_point = random.randint(num_hist, max_valid - 1)
        
        actions, energy_history, grad_norms = self(batch, random_end_point)
        
        gt_actions = act[:, :random_end_point, :]

        loss = self.loss_fn(actions, gt_actions)
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log('train_alpha', self.alpha.item(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('train_horizon', random_end_point - num_hist, on_step=True, on_epoch=True)
        
        if energy_history:
            self.log('train_initial_energy', energy_history[0], on_step=True, on_epoch=True)
            self.log('train_final_energy', energy_history[-1], on_step=True, on_epoch=True)
            self.log('train_energy_reduction', energy_history[0] - energy_history[-1], on_step=True, on_epoch=True)
        
        if grad_norms:
            self.log('train_mean_grad_norm', sum(grad_norms) / len(grad_norms), on_step=True, on_epoch=True)
            self.log('train_max_grad_norm', max(grad_norms), on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step: same as training but with different logging prefix
        """
        obs, act, state = batch

        num_hist = self.world_model.num_hist

        num_frames = obs['visual'].shape[1]
        num_proprio_frames = obs['proprio'].shape[1]
        max_valid = min(num_frames, num_proprio_frames)

        random_end_point = random.randint(num_hist, max_valid - 1)
        
        actions, energy_history, grad_norms = self(batch, random_end_point)
                
        gt_actions = act[:, :random_end_point, :]
        loss = self.loss_fn(actions, gt_actions)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        
        if energy_history:
            self.log('val_initial_energy', energy_history[0], on_step=False, on_epoch=True)
            self.log('val_final_energy', energy_history[-1], on_step=False, on_epoch=True)
            self.log('val_energy_reduction', energy_history[0] - energy_history[-1], on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Add EBM params
        params = list(self.transformer.parameters())
        if self.alpha.requires_grad:
            params.append(self.alpha)
        
        # Remove DINO params (already frozen, but be explicit)
        world_model_param_ids = {id(p) for p in self.world_model.parameters()}
        params = [p for p in params if id(p) not in world_model_param_ids]
        
        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        return optimizer