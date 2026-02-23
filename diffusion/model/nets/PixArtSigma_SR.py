import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    """Sigma backbone with DiTSR-v8 adapter injection logic."""

    def __init__(
        self,
        sparse_inject_ratio: float = 1.0,
        injection_cutoff_layer: int = 25,
        injection_strategy: str = "front_dense",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = len(self.blocks)
        self.injection_cutoff_layer = injection_cutoff_layer

        self._init_injection_strategy(self.depth, mode=injection_strategy, sparse_ratio=sparse_inject_ratio)
        n = len(self.injection_layers)

        self.style_fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.adapter_alpha_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Tanh(),
        )
        self.input_adapter_ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.input_adaln = nn.ModuleList([nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True) for _ in range(n)])
        self.input_res_gate = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(n)])

        for lin in self.input_adaln:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in self.input_res_gate:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _init_injection_strategy(self, depth, mode='front_dense', sparse_ratio=1.0):
        cutoff = min(self.injection_cutoff_layer, depth)
        if mode == 'front_dense':
            layers = list(range(0, min(15, cutoff)))
            layers.extend(list(range(15, min(25, cutoff), 2)))
            self.injection_layers = sorted(set(layers))
            return
        if mode in ('full', 'all'):
            self.injection_layers = list(range(min(depth, cutoff)))
            return

        all_layers = list(range(depth))
        if sparse_ratio < 1.0:
            num_keep = max(1, int(len(all_layers) * sparse_ratio))
            self.injection_layers = [l for l in all_layers[:num_keep] if l < self.injection_cutoff_layer]
        else:
            self.injection_layers = [l for l in all_layers if l < self.injection_cutoff_layer]

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, **kwargs):
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation, base_size=self.base_size
            )
        ).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed
        t = self.t_embedder(timestep)

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

        adapter_features = {}
        style_vec = None
        if adapter_cond is not None and isinstance(adapter_cond, (tuple, list)) and len(adapter_cond) == 2:
            adapter_features, style_vec = adapter_cond

        if style_vec is not None:
            t = t + self.style_fusion_mlp(style_vec.to(self.dtype))

        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        alpha = self.adapter_alpha_mlp(t).view(-1, 1, 1)
        for i, block in enumerate(self.blocks):
            if i in adapter_features and i in self.injection_layers and i < self.injection_cutoff_layer:
                scale_idx = self.injection_layers.index(i)
                feat = adapter_features[i]
                if feat.shape[-2:] != (self.h, self.w):
                    feat = F.interpolate(feat, size=(self.h, self.w), mode='bilinear', align_corners=False)
                feat = feat.flatten(2).transpose(1, 2)

                with torch.cuda.amp.autocast(enabled=False):
                    feat = self.input_adapter_ln(feat.float())
                    adaln_shift, adaln_scale = self.input_adaln[scale_idx](feat).chunk(2, dim=-1)
                layer_alpha = alpha
                x = x + layer_alpha * self.input_res_gate[scale_idx](feat).to(x.dtype)
                x = auto_grad_checkpoint(
                    block,
                    x,
                    y,
                    t0,
                    y_lens,
                    HW=(self.h, self.w),
                    base_size=self.base_size,
                    pe_interpolation=self.pe_interpolation,
                    adaln_shift=adaln_shift,
                    adaln_scale=adaln_scale,
                    adaln_alpha=layer_alpha,
                    **kwargs,
                )
            else:
                x = auto_grad_checkpoint(
                    block,
                    x,
                    y,
                    t0,
                    y_lens,
                    HW=(self.h, self.w),
                    base_size=self.base_size,
                    pe_interpolation=self.pe_interpolation,
                    **kwargs,
                )

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
