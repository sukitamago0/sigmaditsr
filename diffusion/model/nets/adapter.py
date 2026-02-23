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


class MultiLevelAdapterV7(nn.Module):
    """v7 adapter: dynamic per-layer feature dispenser + global style vector."""

    def __init__(self, in_channels: int = 4, hidden_size: int = 1152, base_channels: int = 128, injection_layers_map=None):
        super().__init__()
        self.style_extractor = StyleExtractor(in_channels, hidden_size)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )
        self.body1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            ResBlock(base_channels * 2),
            ResBlock(base_channels * 2),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            ResBlock(base_channels * 8),
            ResBlock(base_channels * 8),
        )

        self.lat0 = nn.Conv2d(base_channels, base_channels, 1)
        self.lat1 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.lat2 = nn.Conv2d(base_channels * 4, base_channels, 1)
        self.lat3 = nn.Conv2d(base_channels * 8, base_channels, 1)

        self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

        self.heads = nn.ModuleDict()
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
            stride = 2 if src_level == 0 else 1
            head = nn.Sequential(
                nn.Conv2d(base_channels, hidden_size, 3, stride=stride, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_size, hidden_size, 1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads[str(lid)] = head

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

    def forward(self, x: torch.Tensor):
        style_vec = self.style_extractor(x)
        f0 = self.stem(x)
        f1 = self.body1(f0)
        f2 = self.body2(f1)
        f3 = self.body3(f2)
        p0, p1, p2, p3 = self._fpn(f0, f1, f2, f3)
        pyramid = {0: p0, 1: p1, 2: p2, 3: p3}
        target_size = p1.shape[-2:]

        outputs = {}
        for layer_str, head in self.heads.items():
            layer_id = int(layer_str)
            lvl = self.layer_to_level[layer_id]
            feat = pyramid[lvl]
            if lvl > 1 and feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            outputs[layer_id] = head(feat)

        return outputs, style_vec


def build_adapter_v7(in_channels=4, hidden_size=1152, injection_layers_map=None):
    return MultiLevelAdapterV7(
        in_channels=in_channels,
        hidden_size=hidden_size,
        base_channels=128,
        injection_layers_map=injection_layers_map,
    )
