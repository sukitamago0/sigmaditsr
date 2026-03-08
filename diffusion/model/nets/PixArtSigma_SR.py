import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    """Sigma backbone with non-invasive DiTSR-style adapter injection."""

    def __init__(
        self,
        sparse_inject_ratio: float = 1.0,
        injection_cutoff_layer: int = 28,
        injection_strategy: str = "three_stage_sr",
        force_null_caption: bool = True,
        injection_r_end: float = 0.1,
        injection_s_min: float = 0.1,
        injection_s_max: float = 1.0,
        injection_init_p: float = 2.0,
        use_csft: bool = True,
        use_style_fusion: bool = False,
        hard_injection_layers=(2, 4, 6, 8, 10, 12),
        transition_injection_layers=tuple(),
        detail_injection_layers=(14, 16, 18, 20, 22, 24),
        **kwargs,
    ):
        # Root-cause alignment for Sigma->SR adaptation:
        # - Sigma checkpoint uses 300 text tokens.
        # - SR script expects direct latent prediction branch (non-sigma head) by default.
        kwargs.setdefault("model_max_length", 300)
        kwargs.setdefault("pred_sigma", False)
        kwargs.setdefault("learn_sigma", False)
        out_channels = kwargs.pop("out_channels", None)
        super().__init__(**kwargs)

        if out_channels is not None and int(out_channels) != int(self.out_channels):
            self.out_channels = int(out_channels)
            head_hidden = self.x_embedder.proj.out_channels
            self.final_layer = T2IFinalLayer(head_hidden, self.patch_size, self.out_channels)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)
        self.depth = len(self.blocks)
        self.hidden_size = self.x_embedder.proj.out_channels
        self.injection_cutoff_layer = injection_cutoff_layer
        self.force_null_caption = force_null_caption
        self.aug_embedder = TimestepEmbedder(self.hidden_size)
        self.hard_injection_layers = sorted(set(int(i) for i in hard_injection_layers))
        self.transition_injection_layers = sorted(set(int(i) for i in transition_injection_layers))
        self.detail_injection_layers = sorted(set(int(i) for i in detail_injection_layers))

        self._init_injection_strategy(self.depth, mode=injection_strategy, sparse_ratio=sparse_inject_ratio)
        self.injection_index_map = {int(layer_id): idx for idx, layer_id in enumerate(self.injection_layers)}
        self.stage_layer_set = {
            "struct": set(l for l in self.injection_layers if l in self.hard_injection_layers),
            "trans": set(l for l in self.injection_layers if l in self.transition_injection_layers),
            "detail": set(l for l in self.injection_layers if l in self.detail_injection_layers),
        }
        self.injection_layer_to_level = self._build_injection_layer_to_level(self.depth)
        self.register_buffer("injection_depth_decay", self._build_injection_depth_decay(depth=self.depth, r_end=float(injection_r_end)), persistent=True)
        n = len(self.injection_layers)
        self.use_csft = bool(use_csft)
        self.use_style_fusion = bool(use_style_fusion)
        self.injection_scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(n)])
        self._init_injection_scales(depth=self.depth, s_max=float(injection_s_max), s_min=float(injection_s_min), p=float(injection_init_p))

        self.style_fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        # Kept for checkpoint/script backward-compatibility (three-window alphas are used in forward).
        self.adapter_alpha_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )
        # Window-specific non-negative alphas (replace global per-sample alpha coupling).
        self.alpha_struct = nn.Parameter(torch.tensor(1.0))
        self.alpha_trans = nn.Parameter(torch.tensor(0.7))
        self.alpha_detail = nn.Parameter(torch.tensor(0.35))
        self.input_adapter_ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.input_adaln = nn.ModuleList([nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True) for _ in range(n)])
        self.input_res_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(n)])
        self.csft_dw = nn.ModuleList([
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, groups=self.hidden_size, bias=True)
            for _ in range(n)
        ])
        self.csft_pw = nn.ModuleList([
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0, bias=True)
            for _ in range(n)
        ])

        for lin in self.input_adaln:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in self.input_res_proj:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # CSFT init policy: zero-impact output + non-zero gradient path.
        # - dw: identity-like init so dw(feat) is non-zero at step0.
        # - pw: zero init so pw(dw(feat)) == 0 at step0 (behavior-preserving).
        for conv in self.csft_dw:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            k_h, k_w = conv.kernel_size
            conv.weight.data[:, :, k_h // 2, k_w // 2] = 1.0
        for conv in self.csft_pw:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

        # Semantic-detail cross-attention (late hooks only), zero-impact at init.
        self.sem_num_heads = 16
        self.sem_head_dim = self.hidden_size // self.sem_num_heads
        self.sem_detail_layers = [l for l in self.detail_injection_layers if 0 <= l < self.depth]
        self.sem_norm = nn.ModuleDict()
        self.sem_q = nn.ModuleDict()
        self.sem_kv = nn.ModuleDict()
        self.sem_out = nn.ModuleDict()
        self.sem_gate = nn.ParameterDict()
        for lid in self.sem_detail_layers:
            k = str(lid)
            self.sem_norm[k] = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
            self.sem_q[k] = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.sem_kv[k] = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=True)
            self.sem_out[k] = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            nn.init.zeros_(self.sem_out[k].weight)
            nn.init.zeros_(self.sem_out[k].bias)
            self.sem_gate[k] = nn.Parameter(torch.zeros(1))

    def load_pretrained_weights_with_zero_init(self, state_dict):
        """Shape-aware checkpoint loading for Sigma base -> SR backbone adaptation.

        Handles deterministic structural differences:
        - x_embedder input channels 4 -> 8 (concat z_t and z_lr)
        - caption token table (120/300 variants)
        - final_layer dimensions when pred_sigma/in_channels differ
        """
        own_state = self.state_dict()
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Never load static position embedding directly.
        state_dict = {k: v for k, v in state_dict.items() if k != "pos_embed"}

        loaded, skipped = 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                skipped += 1
                continue

            src = param.data if isinstance(param, nn.Parameter) else param
            dst = own_state[name]

            # Adapt first conv channels from 4->8 when needed.
            if name == "x_embedder.proj.weight" and src.ndim == 4 and dst.ndim == 4 and src.shape[1] == 4 and dst.shape[1] == 8:
                dst[:, :4, :, :] = src
                dst[:, 4:, :, :] = src * 0.5
                loaded += 1
                continue

            # Text embedding table: load overlapping prefix to support 120<->300 variants.
            if name == "y_embedder.y_embedding" and src.ndim == 2 and dst.ndim == 2 and src.shape[1] == dst.shape[1] and src.shape[0] != dst.shape[0]:
                n = min(src.shape[0], dst.shape[0])
                dst[:n].copy_(src[:n])
                loaded += 1
                continue

            # Final layer mismatch is expected between base and SR heads.
            if name.startswith("final_layer.linear") and src.shape != dst.shape:
                skipped += 1
                continue

            if src.shape == dst.shape:
                dst.copy_(src)
                loaded += 1
            else:
                skipped += 1

        return loaded, skipped

    def _init_injection_strategy(self, depth, mode='front_dense', sparse_ratio=1.0):
        if mode in ('three_stage_sr', 'three_stage', 'sr_windowed'):
            union = sorted(set(self.hard_injection_layers + self.transition_injection_layers + self.detail_injection_layers))
            self.injection_layers = [l for l in union if 0 <= l < min(depth, self.injection_cutoff_layer)]
            return
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

    def _build_injection_depth_decay(self, depth: int, r_end: float = 0.1):
        x = torch.linspace(0, 1, depth)
        decay = r_end + (1 - r_end) * (1 - x)
        return decay

    def _build_injection_layer_to_level(self, depth: int):
        m = {}
        for lid in range(depth):
            if lid < 8:
                lvl = 0
            elif lid < 15:
                lvl = 1
            elif lid < 22:
                lvl = 2
            else:
                lvl = 3
            m[lid] = lvl
        return m

    def _init_injection_scales(self, depth: int, s_max: float = 1.0, s_min: float = 0.1, p: float = 2.0):
        if len(self.injection_layers) == 0:
            return
        denom = float(max(1, depth - 1))
        for i, layer_idx in enumerate(self.injection_layers):
            u = float(layer_idx) / denom
            init_val = s_max - (s_max - s_min) * (u ** p)
            self.injection_scales[i].data.fill_(init_val)

    def _tokens_to_map(self, tokens):
        b, n, c = tokens.shape
        assert n == self.h * self.w
        return tokens.transpose(1, 2).reshape(b, c, self.h, self.w)

    def _map_to_tokens(self, feat):
        return feat.flatten(2).transpose(1, 2)

    def _resolve_stage(self, layer_idx: int) -> str:
        if layer_idx in self.stage_layer_set["struct"]:
            return "struct"
        if layer_idx in self.stage_layer_set["trans"]:
            return "trans"
        return "detail"

    def _window_alpha(self, stage_name: str, dtype, device):
        if stage_name == "struct":
            raw = self.alpha_struct
        elif stage_name == "trans":
            raw = self.alpha_trans
        else:
            raw = self.alpha_detail
        return F.softplus(raw).to(device=device, dtype=dtype).view(1, 1, 1)

    def _time_gates(self, timestep: torch.Tensor, dtype, device):
        t = timestep.float().clamp(0, 999) / 999.0
        g_struct = torch.sigmoid(6.0 * (t - 0.5)).to(device=device, dtype=dtype).view(-1, 1, 1)
        g_sem = torch.sigmoid(6.0 * (0.5 - t)).to(device=device, dtype=dtype).view(-1, 1, 1)
        return g_struct, g_sem

    def _semantic_cross_attn(self, layer_idx: int, x_tokens: torch.Tensor, sem_tokens: torch.Tensor, sem_gate_t: torch.Tensor):
        if sem_tokens is None or layer_idx not in self.sem_detail_layers:
            return x_tokens
        k = str(layer_idx)
        b, n, c = x_tokens.shape
        h = self.sem_norm[k](x_tokens)
        q = self.sem_q[k](h).view(b, n, self.sem_num_heads, self.sem_head_dim).transpose(1, 2)
        sem_n = sem_tokens.shape[1]
        kv = self.sem_kv[k](sem_tokens).view(b, sem_n, 2, self.sem_num_heads, self.sem_head_dim)
        k_t = kv[:, :, 0].transpose(1, 2)
        v_t = kv[:, :, 1].transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k_t, v_t, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, c)
        attn_out = self.sem_out[k](attn_out)
        gate = torch.sigmoid(self.sem_gate[k]).view(1, 1, 1).to(dtype=x_tokens.dtype, device=x_tokens.device)
        return x_tokens + gate * sem_gate_t * attn_out

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
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
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

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

                per_window_alpha = self._window_alpha(stage_name, dtype=x.dtype, device=x.device)
                layer_decay = self.injection_depth_decay[i].to(device=x.device, dtype=x.dtype).view(1, 1, 1)
                g_struct_t, g_sem_t = self._time_gates(timestep=timestep, dtype=x.dtype, device=x.device)
                layer_alpha = F.softplus(self.injection_scales[scale_idx]).to(dtype=x.dtype) * layer_decay * per_window_alpha
                if stage_name in ("struct", "trans"):
                    layer_alpha = layer_alpha * g_struct_t
                res_strength = 1.0 if stage_name == "struct" else (0.35 if stage_name == "trans" else 0.15)
                x = x + (layer_alpha * res_strength) * res.to(x.dtype)

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
                _, g_sem_t = self._time_gates(timestep=timestep, dtype=x.dtype, device=x.device)
                x = self._semantic_cross_attn(i, x, semantic_tokens, g_sem_t)
                continue

            # non-invasive: do not change native block internals
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
            _, g_sem_t = self._time_gates(timestep=timestep, dtype=x.dtype, device=x.device)
            x = self._semantic_cross_attn(i, x, semantic_tokens, g_sem_t)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
