import torch
import torch.nn as nn

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    def __init__(self, force_null_caption: bool = True, adapter_ca_block_ids=None, **kwargs):
        kwargs.setdefault("model_max_length", 300)
        kwargs.setdefault("pred_sigma", False)
        kwargs.setdefault("learn_sigma", False)
        out_channels = kwargs.pop("out_channels", None)
        super().__init__(**kwargs)

        if out_channels is not None and int(out_channels) != int(self.out_channels):
            self.out_channels = int(out_channels)
            head_hidden = self.x_embedder.proj.out_channels
            self.final_layer = T2IFinalLayer(head_hidden, self.patch_size, self.out_channels)
            nn.init.trunc_normal_(self.final_layer.linear.weight, std=0.02)
            nn.init.constant_(self.final_layer.linear.bias, 0)

        self.depth = len(self.blocks)
        self.hidden_size = self.x_embedder.proj.out_channels
        self.force_null_caption = bool(force_null_caption)
        self.aug_embedder = TimestepEmbedder(self.hidden_size)

        # sparse decoupled adapter cross-attention branch
        if adapter_ca_block_ids is None:
            if self.depth == 28:
                self.adapter_ca_block_ids = [14, 18, 22, 26]
            else:
                ratios = [0.50, 0.64, 0.78, 0.93]
                cand = sorted(set(min(self.depth - 1, max(0, int(round((self.depth - 1) * r)))) for r in ratios))
                while len(cand) < 4:
                    cand.append(cand[-1])
                self.adapter_ca_block_ids = cand[:4]
        else:
            self.adapter_ca_block_ids = [int(x) for x in adapter_ca_block_ids]

        head_nums = []
        for bid in self.adapter_ca_block_ids:
            b = self.blocks[bid]
            h = getattr(getattr(b, "attn", None), "num_heads", None)
            if h is None:
                raise RuntimeError(f"Cannot resolve attention heads from block {bid}")
            head_nums.append(int(h))
        if len(set(head_nums)) != 1:
            raise RuntimeError(f"Inconsistent num_heads across selected blocks: {head_nums}")
        self.adapter_ca_num_heads = head_nums[0]

        self.adapter_ca_norm_q = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in self.adapter_ca_block_ids])
        self.adapter_ca_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_size, self.adapter_ca_num_heads, batch_first=True) for _ in self.adapter_ca_block_ids
        ])
        self.adapter_ca_out = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in self.adapter_ca_block_ids])
        # use logit gate for small but non-zero initial impact and non-dead branch
        self.adapter_ca_gate = nn.Parameter(torch.full((len(self.adapter_ca_block_ids),), -4.0))
        self._adapter_ca_index = {bid: i for i, bid in enumerate(self.adapter_ca_block_ids)}

        for out in self.adapter_ca_out:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)

        self._shape_logged = False
        print(
            f"[PixArtShapeCheck] depth={self.depth}, hidden={self.hidden_size}, num_heads={self.adapter_ca_num_heads}, "
            f"adapter_ca_block_ids={self.adapter_ca_block_ids}"
        )

    def _apply_adapter_ca(self, x: torch.Tensor, block_id: int, memory_tokens: torch.Tensor):
        idx = self._adapter_ca_index[block_id]
        q = self.adapter_ca_norm_q[idx](x)
        delta, _ = self.adapter_ca_layers[idx](q, memory_tokens, memory_tokens, need_weights=False)
        delta = self.adapter_ca_out[idx](delta)
        gate = torch.sigmoid(self.adapter_ca_gate[idx]).to(dtype=x.dtype)
        x = x + gate * delta.to(dtype=x.dtype)
        return x

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation, base_size=self.base_size)
        ).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed
        t = self.t_embedder(timestep)
        if aug_level is not None:
            t = t + self.aug_embedder(aug_level.to(self.dtype))

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

        memory_tokens = None
        if isinstance(adapter_cond, dict):
            memory_tokens = adapter_cond.get("memory_tokens", None)

        if memory_tokens is not None:
            memory_tokens = memory_tokens.to(dtype=x.dtype)
            if not self._shape_logged:
                print(f"[PixArtShapeCheck] x_tokens={tuple(x.shape)} memory_tokens={tuple(memory_tokens.shape)}")
                self._shape_logged = True

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

        for i, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                HW=(self.h, self.w),
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                adaln_shift=None,
                adaln_scale=None,
                adaln_alpha=None,
                **kwargs,
            )
            if (memory_tokens is not None) and (i in self._adapter_ca_index):
                x = self._apply_adapter_ca(x, i, memory_tokens)

        self._last_adapter_ca_gates = torch.sigmoid(self.adapter_ca_gate.detach().float()).cpu()

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
