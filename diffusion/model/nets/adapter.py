import math
import torch
import torch.nn as nn

from diffusion.model.nets.srconvnet_blocks import SRConvNetBlock


class PerScaleResamplerBlock(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, ffn_expansion: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim),
        )

    def forward(self, latent_q: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(latent_q)
        kv = self.norm_kv(kv_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = latent_q + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MultiScaleMemoryResampler(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, depth: int = 2):
        super().__init__()
        self.blocks_f2 = nn.ModuleList([PerScaleResamplerBlock(dim=dim, heads=heads, ffn_expansion=4) for _ in range(depth)])
        self.blocks_f3 = nn.ModuleList([PerScaleResamplerBlock(dim=dim, heads=heads, ffn_expansion=4) for _ in range(depth)])
        self.blocks_f4 = nn.ModuleList([PerScaleResamplerBlock(dim=dim, heads=heads, ffn_expansion=4) for _ in range(depth)])

    def forward(self, q2: torch.Tensor, q3: torch.Tensor, q4: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor):
        for blk in self.blocks_f2:
            q2 = blk(q2, t2)
        for blk in self.blocks_f3:
            q3 = blk(q3, t3)
        for blk in self.blocks_f4:
            q4 = blk(q4, t4)
        return q2, q3, q4


class SRConvNetLSAAdapter(nn.Module):
    def __init__(self, hidden_size: int = 1152):
        super().__init__()
        self.hidden_size = int(hidden_size)

        # keep existing backbone
        self.stem = nn.Conv2d(3, 64, 3, padding=1)
        self.stage1 = nn.Sequential(SRConvNetBlock(64), SRConvNetBlock(64))
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(SRConvNetBlock(128), SRConvNetBlock(128))
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(SRConvNetBlock(256), SRConvNetBlock(256), SRConvNetBlock(256), SRConvNetBlock(256))
        self.stage4 = nn.Sequential(SRConvNetBlock(256), SRConvNetBlock(256))

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * (64 + 128 + 256 + 256)),
        )

        self.proj2 = nn.Conv2d(128, 256, 1)
        self.proj3 = nn.Conv2d(256, 256, 1)
        self.proj4 = nn.Conv2d(256, 256, 1)

        # legacy heads: kept for checkpoint compatibility only (unused in new forward path)
        self.out_proj_early = nn.Conv2d(512, self.hidden_size, 1)
        self.out_proj_mid = nn.Conv2d(512, self.hidden_size, 1)
        self.out_proj_late = nn.Conv2d(768, self.hidden_size, 1)

        # new memory path
        self.mem_proj_f2 = nn.Conv2d(128, 512, 1)
        self.mem_proj_f3 = nn.Conv2d(256, 512, 1)
        self.mem_proj_f4 = nn.Conv2d(256, 512, 1)

        self.pos_proj = nn.Linear(4, 512)
        self.scale_embed = nn.Parameter(torch.zeros(3, 512))

        self.q2 = nn.Parameter(torch.randn(64, 512) * 0.02)
        self.q3 = nn.Parameter(torch.randn(32, 512) * 0.02)
        self.q4 = nn.Parameter(torch.randn(16, 512) * 0.02)

        self.resampler = MultiScaleMemoryResampler(dim=512, heads=8, depth=2)
        self.memory_out_proj = nn.Linear(512, self.hidden_size)
        self.memory_ln = nn.LayerNorm(self.hidden_size)

        for m in [
            self.proj2,
            self.proj3,
            self.proj4,
            self.out_proj_early,
            self.out_proj_mid,
            self.out_proj_late,
            self.mem_proj_f2,
            self.mem_proj_f3,
            self.mem_proj_f4,
        ]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.pos_proj.weight)
        nn.init.zeros_(self.pos_proj.bias)
        nn.init.xavier_uniform_(self.memory_out_proj.weight)
        nn.init.zeros_(self.memory_out_proj.bias)

        self._shape_logged = False

    @staticmethod
    def _film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return (1.0 + gamma[:, :, None, None]) * feat + beta[:, :, None, None]

    def _build_pos_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        rr = torch.sqrt(xx * xx + yy * yy)
        ang = torch.atan2(yy, xx) / math.pi
        pos = torch.stack([xx, yy, rr, ang], dim=-1).view(1, h * w, 4).repeat(b, 1, 1)
        return self.pos_proj(pos.float()).to(dtype=feat.dtype)

    def _tokenize_with_pos_scale(self, feat: torch.Tensor, scale_id: int) -> torch.Tensor:
        tok = feat.flatten(2).transpose(1, 2)
        pos = self._build_pos_tokens(feat)
        tok = tok + pos + self.scale_embed[scale_id].view(1, 1, -1).to(dtype=tok.dtype)
        return tok

    def forward(self, lr_small: torch.Tensor, t_embed: torch.Tensor = None):
        f1 = self.stage1(self.stem(lr_small))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(f3)

        if t_embed is not None:
            tb = self.time_mlp(t_embed)
            splits = [64, 128, 256, 256, 64, 128, 256, 256]
            g1, g2, g3, g4, b1, b2, b3, b4 = tb.split(splits, dim=-1)
            f1 = self._film(f1, g1, b1)
            f2 = self._film(f2, g2, b2)
            f3 = self._film(f3, g3, b3)
            f4 = self._film(f4, g4, b4)

        # channel checks from real implementation
        if not self._shape_logged:
            print(
                f"[AdapterShapeCheck] f2={tuple(f2.shape)} f3={tuple(f3.shape)} f4={tuple(f4.shape)} "
                f"| C2={f2.shape[1]} C3={f3.shape[1]} C4={f4.shape[1]}"
            )

        tf2 = self._tokenize_with_pos_scale(self.mem_proj_f2(f2), scale_id=0)
        tf3 = self._tokenize_with_pos_scale(self.mem_proj_f3(f3), scale_id=1)
        tf4 = self._tokenize_with_pos_scale(self.mem_proj_f4(f4), scale_id=2)

        q2 = self.q2.unsqueeze(0).expand(tf2.shape[0], -1, -1)
        q3 = self.q3.unsqueeze(0).expand(tf3.shape[0], -1, -1)
        q4 = self.q4.unsqueeze(0).expand(tf4.shape[0], -1, -1)
        m2, m3, m4 = self.resampler(q2, q3, q4, tf2, tf3, tf4)

        memory = torch.cat([m2, m3, m4], dim=1)
        memory = self.memory_ln(self.memory_out_proj(memory))

        if not self._shape_logged:
            print(
                f"[AdapterShapeCheck] tf2={tuple(tf2.shape)} tf3={tuple(tf3.shape)} tf4={tuple(tf4.shape)} "
                f"-> memory={tuple(memory.shape)}"
            )
            self._shape_logged = True

        return {
            "memory_tokens": memory,
            "memory_meta": {"n2": 64, "n3": 32, "n4": 16},
        }


def build_adapter_v8(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)
