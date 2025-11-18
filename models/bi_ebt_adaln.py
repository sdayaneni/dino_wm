import torch
import torch.nn as nn
import numpy as np
import math
import pytorch_lightning as L
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from models.model_utils import *


# adapted from https://arxiv.org/pdf/2212.09748 and https://github.com/facebookresearch/DiT :)))
# major change is DiT -> EBT as well as changing output layer to scalar

class EBTBlock(L.LightningModule):
    """
    A EBT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        inside_attn = modulate(self.norm1(x), shift_msa, scale_msa)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False): #TODO may want to turn this off for inference eventually? unsure; upgrade to use newer version
            attn_results = self.attn(inside_attn) # needed to set this as regular sdpa from pt didnt support higher order gradients
        x = x + gate_msa.unsqueeze(1) * attn_results
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class EnergyFinalLayer(nn.Module):
    """
    Energy Final layer that outputs a scalar energy value per sample.
    """
    def __init__(self, input_dim, linear_then_mean=False):
        super().__init__()
        self.linear_then_mean = linear_then_mean
        self.energy_head = nn.Sequential( # this helps to match the parameter count of standard DiTs better
            nn.Linear(input_dim, input_dim * 2),
            nn.SiLU(),
            nn.Linear(input_dim * 2, 1, False) # need no bias here since wont have a grad if doing reconstruction loss
        )
        self.norm_final = nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, 2 * input_dim, bias=True)
        )

    def forward(self, x, c):
        # x: (N, T, D) -> energy: (N, 1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        if self.linear_then_mean:
            # Apply linear layers first, then mean across T dim
            x_transformed = self.energy_head(x)  # (N, T, 1)
            energy = x_transformed.mean(dim=1)   # (N, 1)
        else:
            # Original: mean first across T dim, then linear layers; this tends to work better
            x_pooled = x.mean(dim=1)  # (N, D)
            energy = self.energy_head(x_pooled)  # (N, 1)
        return energy


class EBT(nn.Module): # similar to DiT but with no time embedder and gives single scalar for each patch not a noise prediction
    """
    Energy Based model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        is_time_conditioned=False,
        linear_then_mean=False
    ):
        super().__init__()
        self.in_channels = in_channels # no notion of out channels since is ebm
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.channel_height = input_size
        self.channel_width = input_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.is_time_conditioned = is_time_conditioned
        if is_time_conditioned:
            self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches

        # For now, using learned positional encoding TODO: add sin/cos as well
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(self.pos_embed.data) 

        self.blocks = nn.ModuleList([
            EBTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = EnergyFinalLayer(hidden_size, linear_then_mean=linear_then_mean)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)) # TODO: implement later
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in EBT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        for m in self.final_layer.energy_head.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight) #NOTE use xavier uniform since with zero init EBT cannot learn anything
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0) # turned off bias for final output for ebm

    def forward(self, x, y, pos_embed, t=None, return_patches = False): # optional time conditioning since is an EBT, return patches is for downstream models not using final layer etc
        """
        Forward pass of EBT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        y: (N,D) tensor of conditions (not necessarily labels)
        t: (N,) [optional] tensor of diffusion timesteps
        """
        if self.is_time_conditioned:
            assert t is not None
        else:
            assert t is None
        
        # Removed patch embed
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        x = x + pos_embed

        c = y                                # (N, D)

        if t is not None:
            t = self.t_embedder(t)                   # (N, D)
            c = c + t  

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        if return_patches:
            return x
        x = self.final_layer(x, c)                # (N, 1), no need to unpatchify since are using EBM, backward pass naturally 'unpatchifies'
        return x
