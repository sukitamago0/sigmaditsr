import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PatchEmbed
from diffusion.model.nets.PixArtSigma_SR import PixArtSigmaSR
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


@MODELS.register_module()
class PixArtSigmaSRDualStream(PixArtSigmaSR):
    """Sigma SR with dual-stream LR<->noise interaction in late layers.

    Zero-impact initialization:
    - dual_out projections are zero-init
    - per-layer dual_gate is zero-init
    """

    def __init__(
        self,
        dualstream_enabled: bool = False,
        cross_attn_start_layer: int = 16,
        dual_num_heads: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dualstream_enabled = dualstream_enabled
        self.cross_attn_start_layer = int(cross_attn_start_layer)
        self.dual_num_heads = int(dual_num_heads)
        self.dual_head_dim = self.hidden_size // self.dual_num_heads
        if self.hidden_size % self.dual_num_heads != 0:
            raise ValueError(f"hidden_size={self.hidden_size} not divisible by dual_num_heads={self.dual_num_heads}")

        self.lr_embedder = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=4,
            embed_dim=self.hidden_size,
            flatten=True,
            bias=True,
        )

        active_layers = [i for i in range(self.depth) if i >= self.cross_attn_start_layer]
        self.dual_active_layers = active_layers

        self.dual_norm = nn.ModuleDict()
        self.dual_q = nn.ModuleDict()
        self.dual_kv = nn.ModuleDict()
        self.dual_out = nn.ModuleDict()
        self.dual_gate = nn.ParameterDict()

        for lid in active_layers:
            key = str(lid)
            self.dual_norm[key] = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
            self.dual_q[key] = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.dual_kv[key] = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=True)
            self.dual_out[key] = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            nn.init.zeros_(self.dual_out[key].weight)
            nn.init.zeros_(self.dual_out[key].bias)
            self.dual_gate[key] = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def init_lr_embedder_from_x_embedder(self):
        """Copy-init LR embedder from x_embedder first 4 channels."""
        if hasattr(self.x_embedder, "proj") and hasattr(self.lr_embedder, "proj"):
            xw = self.x_embedder.proj.weight.detach()
            lw = self.lr_embedder.proj.weight
            if xw.ndim == 4 and lw.ndim == 4 and xw.shape[1] >= 4 and lw.shape[1] == 4:
                lw.copy_(xw[:, :4, :, :])
                if self.x_embedder.proj.bias is not None and self.lr_embedder.proj.bias is not None:
                    self.lr_embedder.proj.bias.copy_(self.x_embedder.proj.bias.detach())

    def _dual_cross_attn(self, layer_idx: int, x_tokens: torch.Tensor, lr_tokens: torch.Tensor) -> torch.Tensor:
        if (not self.dualstream_enabled) or (layer_idx not in self.dual_active_layers):
            return x_tokens
        k = str(layer_idx)
        b, n, c = x_tokens.shape

        h = self.dual_norm[k](x_tokens)
        q = self.dual_q[k](h).view(b, n, self.dual_num_heads, self.dual_head_dim).transpose(1, 2)
        kv = self.dual_kv[k](lr_tokens).view(b, n, 2, self.dual_num_heads, self.dual_head_dim)
        k_t = kv[:, :, 0].transpose(1, 2)
        v_t = kv[:, :, 1].transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k_t, v_t, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, c)
        attn_out = self.dual_out[k](attn_out)

        gate = torch.sigmoid(self.dual_gate[k]).view(1, 1, 1)
        return x_tokens + gate * attn_out

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        if x.shape[1] >= 8:
            # S2D refactor: main backbone only consumes z_t; ignore concatenated z_lr if present.
            x = x[:, :4]
        z_lr = x[:, :4]

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation, base_size=self.base_size
            )
        ).unsqueeze(0).to(x.device).to(self.dtype)

        x_tokens = self.x_embedder(x[:, :4]) + pos_embed
        lr_tokens = self.lr_embedder(z_lr.to(self.dtype)) + pos_embed
        if lr_tokens.shape[1] != x_tokens.shape[1]:
            raise RuntimeError(f"Token mismatch: lr_tokens={lr_tokens.shape}, x_tokens={x_tokens.shape}")

        t = self.t_embedder(timestep)
        if aug_level is not None:
            t = t + self.aug_embedder(aug_level.to(self.dtype))

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

        adapter_features = {}
        adapter_structure_features = {}
        adapter_detail_features = {}
        adapter_gates = {}
        semantic_tokens = None
        style_vec = None
        if adapter_cond is not None and isinstance(adapter_cond, (tuple, list)):
            if len(adapter_cond) == 2:
                adapter_features, style_vec = adapter_cond
            elif len(adapter_cond) >= 3:
                adapter_features, style_vec, adapter_gates = adapter_cond[:3]
            if len(adapter_cond) >= 4:
                semantic_tokens = adapter_cond[3]

        if isinstance(adapter_features, dict) and ("structure" in adapter_features or "detail" in adapter_features):
            adapter_structure_features = adapter_features.get("structure", {})
            adapter_detail_features = adapter_features.get("detail", {})
            if semantic_tokens is None:
                semantic_tokens = adapter_features.get("semantic_tokens", None)
        else:
            adapter_structure_features = adapter_features if isinstance(adapter_features, dict) else {}
            adapter_detail_features = adapter_features if isinstance(adapter_features, dict) else {}

        if style_vec is not None and self.use_style_fusion:
            t = t + self.style_fusion_mlp(style_vec.to(self.dtype))

        t0 = self.t_block(t)

        if force_drop_ids is None and self.force_null_caption:
            force_drop_ids = torch.ones(bs, device=y.device, dtype=torch.long)
        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x_tokens.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x_tokens.shape[-1])

        self._last_adapter_alpha = {
            "struct": float(F.softplus(self.alpha_struct).detach().cpu()),
            "trans": float(F.softplus(self.alpha_trans).detach().cpu()),
            "detail": float(F.softplus(self.alpha_detail).detach().cpu()),
        }
        active_injection_layers = getattr(self, "active_injection_layers", None)
        active_injection_layers = set(active_injection_layers) if active_injection_layers is not None else None
        for i, block in enumerate(self.blocks):
            stage_name = self._resolve_stage(i)
            src_features = adapter_detail_features if stage_name == "detail" else adapter_structure_features
            layer_enabled = (active_injection_layers is None) or (i in active_injection_layers)
            if layer_enabled and i in src_features and i in self.injection_index_map and i < self.injection_cutoff_layer:
                scale_idx = self.injection_index_map[i]
                feat = src_features[i]
                if feat.shape[-2:] != (self.h, self.w):
                    feat = F.interpolate(feat, size=(self.h, self.w), mode='bilinear', align_corners=False)
                if self.use_csft:
                    feat = feat + self.csft_pw[scale_idx](self.csft_dw[scale_idx](feat))
                gate = adapter_gates.get(i, None) if isinstance(adapter_gates, dict) else None
                if gate is not None:
                    if gate.shape[-2:] != (self.h, self.w):
                        gate = F.interpolate(gate, size=(self.h, self.w), mode='bilinear', align_corners=False)
                    if gate.shape[1] != 1:
                        gate = gate.mean(dim=1, keepdim=True)
                    gate = gate.clamp(0.0, 1.0)

                feat = feat.flatten(2).transpose(1, 2)

                with torch.cuda.amp.autocast(enabled=False):
                    feat = self.input_adapter_ln(feat.float())
                    adaln_shift, adaln_scale = self.input_adaln[scale_idx](feat).chunk(2, dim=-1)
                res = self.input_res_proj[scale_idx](feat)

                if gate is not None:
                    gate_tokens = gate.flatten(2).transpose(1, 2).to(feat.dtype)
                    adaln_shift = adaln_shift * gate_tokens
                    adaln_scale = adaln_scale * gate_tokens
                    res = res * gate_tokens

                per_window_alpha = self._window_alpha(stage_name, dtype=x_tokens.dtype, device=x_tokens.device)
                layer_decay = self.injection_depth_decay[i].to(device=x_tokens.device, dtype=x_tokens.dtype).view(1, 1, 1)
                g_struct_t, _ = self._time_gates(timestep=timestep, dtype=x_tokens.dtype, device=x_tokens.device)
                layer_alpha = F.softplus(self.injection_scales[scale_idx]).to(dtype=x_tokens.dtype) * layer_decay * per_window_alpha
                if stage_name in ("struct", "trans"):
                    layer_alpha = layer_alpha * g_struct_t
                res_strength = 1.0 if stage_name == "struct" else (0.35 if stage_name == "trans" else 0.15)
                x_tokens = x_tokens + (layer_alpha * res_strength) * res.to(x_tokens.dtype)

                x_tokens = auto_grad_checkpoint(
                    block,
                    x_tokens,
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
                x_tokens = auto_grad_checkpoint(
                    block,
                    x_tokens,
                    y,
                    t0,
                    y_lens,
                    HW=(self.h, self.w),
                    base_size=self.base_size,
                    pe_interpolation=self.pe_interpolation,
                    **kwargs,
                )

            semantic_allowed = bool(getattr(self, "enable_semantic_detail", True)) and ((active_injection_layers is None) or (i in active_injection_layers))
            if semantic_allowed:
                _, g_sem_t = self._time_gates(timestep=timestep, dtype=x_tokens.dtype, device=x_tokens.device)
                x_tokens = self._semantic_cross_attn(i, x_tokens, semantic_tokens, g_sem_t)
            x_tokens = self._dual_cross_attn(i, x_tokens, lr_tokens)

        x_tokens = self.final_layer(x_tokens, t)
        x_tokens = self.unpatchify(x_tokens)
        return x_tokens


@MODELS.register_module()
def PixArtSigmaSRDualStream_XL_2(**kwargs):
    return PixArtSigmaSRDualStream(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
