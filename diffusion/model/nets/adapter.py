import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class StyleExtractor(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 1152):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = torch.flatten(feat, 1)
        return self.proj(feat)


class MultiLevelAdapterV8(nn.Module):
    """v8 adapter: dual-branch (RGB + structure) with per-layer feat/gate heads."""

    def __init__(self, in_channels: int = 6, hidden_size: int = 1152, base_channels: int = 128, injection_layers_map=None):
        super().__init__()
        if in_channels not in (4, 6):
            raise ValueError(
                f"MultiLevelAdapterV8 expects 4-channel (legacy) or 6-channel (rgb+gray/sobel/lap) input, got {in_channels}"
            )
        self.in_channels = int(in_channels)

        self.style_extractor = StyleExtractor(in_channels, hidden_size)
        c_half = base_channels // 2

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, c_half, 3, padding=1),
            ResBlock(c_half),
            ResBlock(c_half),
        )
        self.struct_stem = nn.Sequential(
            nn.Conv2d(3, c_half, 3, padding=1),
            ResBlock(c_half),
            ResBlock(c_half),
        )
        self.fuse0 = nn.Sequential(nn.Conv2d(base_channels, base_channels, 1), nn.SiLU())

        self.rgb_body1 = nn.Sequential(
            nn.Conv2d(c_half, base_channels, 3, stride=2, padding=1),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )
        self.struct_body1 = nn.Sequential(
            nn.Conv2d(c_half, base_channels, 3, stride=2, padding=1),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )
        self.fuse1 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 2, 1), nn.SiLU())

        self.rgb_body2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            ResBlock(base_channels * 2),
            ResBlock(base_channels * 2),
        )
        self.struct_body2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            ResBlock(base_channels * 2),
            ResBlock(base_channels * 2),
        )
        self.fuse2 = nn.Sequential(nn.Conv2d(base_channels * 4, base_channels * 4, 1), nn.SiLU())

        self.rgb_body3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4),
        )
        self.struct_body3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4),
        )
        self.fuse3 = nn.Sequential(nn.Conv2d(base_channels * 8, base_channels * 8, 1), nn.SiLU())

        self.lat0 = nn.Conv2d(base_channels, base_channels, 1)
        self.lat1 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.lat2 = nn.Conv2d(base_channels * 4, base_channels, 1)
        self.lat3 = nn.Conv2d(base_channels * 8, base_channels, 1)

        self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

        self.heads = nn.ModuleDict()
        self.gate_heads = nn.ModuleDict()
        self.detail_heads = nn.ModuleDict()
        self.layer_to_level = {}
        if isinstance(injection_layers_map, dict):
            self.injection_layers_map = sorted([int(k) for k in injection_layers_map.keys()])
            self.layer_to_level = {int(k): int(v) for k, v in injection_layers_map.items()}
        else:
            self.injection_layers_map = sorted(injection_layers_map) if injection_layers_map else list(range(21))

        for lid in self.injection_layers_map:
            if int(lid) not in self.layer_to_level:
                if lid < 8:
                    src_level = 0
                elif lid < 15:
                    src_level = 1
                elif lid < 22:
                    src_level = 2
                else:
                    src_level = 3
                self.layer_to_level[int(lid)] = src_level

            src_level = int(self.layer_to_level[int(lid)])
            # Heads run after feature maps are aligned to a unified target_size in forward.
            # Keep stride=1 for all layers to avoid accidental extra downsampling.
            stride = 1
            head = nn.Sequential(
                nn.Conv2d(base_channels, hidden_size, 3, stride=stride, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_size, hidden_size, 1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads[str(lid)] = head

            detail_head = nn.Sequential(
                nn.Conv2d(base_channels, hidden_size, 3, stride=stride, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_size, hidden_size, 1),
            )
            nn.init.zeros_(detail_head[-1].weight)
            nn.init.zeros_(detail_head[-1].bias)
            self.detail_heads[str(lid)] = detail_head

            gate_head = nn.Sequential(
                nn.Conv2d(base_channels, base_channels // 2, 3, stride=stride, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels // 2, 1, 1),
            )
            nn.init.zeros_(gate_head[-1].weight)
            nn.init.zeros_(gate_head[-1].bias)
            self.gate_heads[str(lid)] = gate_head

    def _fpn(self, f0, f1, f2, f3):
        p3 = self.lat3(f3)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.lat0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.refine0(p0)
        p1 = self.refine1(p1)
        p2 = self.refine2(p2)
        p3 = self.refine3(p3)
        return p0, p1, p2, p3

    def forward(self, x: torch.Tensor, return_style: bool = True):
        rgb = x[:, :3]
        if self.in_channels >= 6:
            struct = x[:, 3:6]
        else:
            struct = x[:, 3:4].repeat(1, 3, 1, 1)

        style_vec = self.style_extractor(x) if return_style else None

        r0 = self.rgb_stem(rgb)
        s0 = self.struct_stem(struct)
        f0 = self.fuse0(torch.cat([r0, s0], dim=1))

        r1 = self.rgb_body1(r0)
        s1 = self.struct_body1(s0)
        f1 = self.fuse1(torch.cat([r1, s1], dim=1))

        r2 = self.rgb_body2(r1)
        s2 = self.struct_body2(s1)
        f2 = self.fuse2(torch.cat([r2, s2], dim=1))

        r3 = self.rgb_body3(r2)
        s3 = self.struct_body3(s2)
        f3 = self.fuse3(torch.cat([r3, s3], dim=1))

        p0, p1, p2, p3 = self._fpn(f0, f1, f2, f3)
        pyramid = {0: p0, 1: p1, 2: p2, 3: p3}
        target_size = p2.shape[-2:]
        edge_prior = torch.mean(torch.abs(struct), dim=1, keepdim=True)
        edge_prior = F.interpolate(edge_prior, size=target_size, mode="bilinear", align_corners=False)
        edge_prior = edge_prior / (edge_prior.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        edge_prior_logits = (edge_prior - 0.5) * 4.0

        # Texture prior for detail-stage gating (avoid over-biasing detail gates to hard edges only).
        rgb_luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        rgb_luma = F.interpolate(rgb_luma, size=target_size, mode="bilinear", align_corners=False)
        local_mean = F.avg_pool2d(rgb_luma, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((rgb_luma - local_mean) ** 2, kernel_size=3, stride=1, padding=1)
        texture_prior = torch.sqrt(local_var + 1e-6)
        texture_prior = texture_prior / (texture_prior.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        texture_prior_logits = (texture_prior - 0.5) * 3.0

        structure_outputs = {}
        detail_outputs = {}
        gates = {}
        for layer_str, head in self.heads.items():
            layer_id = int(layer_str)
            lvl = self.layer_to_level[layer_id]
            feat = pyramid[lvl]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            if layer_id <= 17:
                structure_outputs[layer_id] = head(feat)
                prior_logits = edge_prior_logits
            else:
                detail_outputs[layer_id] = self.detail_heads[layer_str](feat)
                prior_logits = 0.35 * edge_prior_logits + 0.65 * texture_prior_logits
            gates[layer_id] = torch.sigmoid(prior_logits + self.gate_heads[layer_str](feat))

        outputs = {
            "structure": structure_outputs,
            "detail": detail_outputs,
        }
        return outputs, style_vec, gates


def build_adapter_v8(in_channels=6, hidden_size=1152, injection_layers_map=None):
    return MultiLevelAdapterV8(
        in_channels=in_channels,
        hidden_size=hidden_size,
        base_channels=128,
        injection_layers_map=injection_layers_map,
    )


def build_adapter_v7(in_channels=6, hidden_size=1152, injection_layers_map=None):
    """Backward-compatible alias for legacy callsites."""
    return build_adapter_v8(
        in_channels=in_channels,
        hidden_size=hidden_size,
        injection_layers_map=injection_layers_map,
    )


# Backward-compatible class alias for old config strings/checkpoints.
MultiLevelAdapterV7 = MultiLevelAdapterV8
