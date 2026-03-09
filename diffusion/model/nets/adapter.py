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


class FrozenSemanticTokenizer(nn.Module):
    """Frozen visual-statistics tokenizer (not a semantic encoder).

    Uses fixed Sobel/Laplacian + low-frequency maps followed by adaptive pooling.
    No trainable parameters by design.
    """

    def __init__(self, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = int(num_tokens)

    def forward(self, img_m11: torch.Tensor, hidden_size: int) -> torch.Tensor:
        # img_m11: [B,3,H,W] in [-1,1]
        img01 = (img_m11.float() + 1.0) * 0.5
        gray = 0.299 * img01[:, 0:1] + 0.587 * img01[:, 1:2] + 0.114 * img01[:, 2:3]

        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=gray.device, dtype=gray.dtype)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=gray.device, dtype=gray.dtype)
        lap = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=gray.device, dtype=gray.dtype)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap_abs = F.conv2d(gray, lap, padding=1).abs()
        low = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)

        sem_map = torch.cat([gray, low, grad, lap_abs], dim=1)
        side = int(self.num_tokens ** 0.5)
        pooled = F.adaptive_avg_pool2d(sem_map, output_size=(side, side))
        tokens = pooled.flatten(2).transpose(1, 2)  # [B,T,4]

        repeat = (hidden_size + tokens.shape[-1] - 1) // tokens.shape[-1]
        tokens = tokens.repeat(1, 1, repeat)[..., :hidden_size]
        return tokens


class MultiLevelAdapterV8(nn.Module):
    """Structure-only adapter + semantic token emitter (S2D decoupled)."""

    def __init__(self, in_channels: int = 4, hidden_size: int = 1152, base_channels: int = 128, injection_layers_map=None):
        super().__init__()
        if in_channels not in (4,):
            raise ValueError(f"MultiLevelAdapterV8 (S2D) expects 4-channel structural input, got {in_channels}")
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)

        self.struct_stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )
        self.struct_body1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            ResBlock(base_channels * 2),
            ResBlock(base_channels * 2),
        )
        self.struct_body2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4),
        )
        self.struct_body3 = nn.Sequential(
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
        self.gate_heads = nn.ModuleDict()
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

            head = nn.Sequential(
                nn.Conv2d(base_channels, hidden_size, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_size, hidden_size, 1),
            )
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head[-1].bias)
            self.heads[str(lid)] = head

            gate_head = nn.Sequential(
                nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels // 2, 1, 1),
            )
            nn.init.zeros_(gate_head[-1].weight)
            nn.init.zeros_(gate_head[-1].bias)
            self.gate_heads[str(lid)] = gate_head

        self.semantic_tokenizer = FrozenSemanticTokenizer(num_tokens=64)

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

    def forward(self, x: torch.Tensor, sem_image: torch.Tensor = None, return_style: bool = True):
        style_vec = None
        f0 = self.struct_stem(x)
        f1 = self.struct_body1(f0)
        f2 = self.struct_body2(f1)
        f3 = self.struct_body3(f2)

        p0, p1, p2, p3 = self._fpn(f0, f1, f2, f3)
        pyramid = {0: p0, 1: p1, 2: p2, 3: p3}
        target_size = p2.shape[-2:]

        edge_prior = torch.mean(torch.abs(x[:, 2:4]), dim=1, keepdim=True)
        edge_prior = F.interpolate(edge_prior, size=target_size, mode="bilinear", align_corners=False)
        edge_prior = edge_prior / (edge_prior.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        edge_prior_logits = (edge_prior - 0.5) * 4.0

        structure_outputs = {}
        detail_outputs = {}
        gates = {}
        for layer_str, head in self.heads.items():
            layer_id = int(layer_str)
            lvl = self.layer_to_level[layer_id]
            feat = pyramid[lvl]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            if layer_id <= 12:
                structure_outputs[layer_id] = head(feat)
            gates[layer_id] = torch.sigmoid(edge_prior_logits + self.gate_heads[layer_str](feat))

        if sem_image is None:
            sem_image = x[:, :3] * 2.0 - 1.0 if x.shape[1] >= 3 else x.repeat(1, 3, 1, 1) * 2.0 - 1.0
        sem_tokens = self.semantic_tokenizer(sem_image, hidden_size=self.hidden_size)

        # S2D note: detail_outputs intentionally stays empty.
        # Late-detail control comes from token cross-attention (semantic_tokens),
        # not from LR-local detail feature maps.
        outputs = {
            "structure": structure_outputs,
            "detail": detail_outputs,
            "semantic_tokens": sem_tokens,
        }
        return outputs, style_vec, gates


def build_adapter_v8(in_channels=4, hidden_size=1152, injection_layers_map=None):
    return MultiLevelAdapterV8(
        in_channels=in_channels,
        hidden_size=hidden_size,
        base_channels=128,
        injection_layers_map=injection_layers_map,
    )


def build_adapter_v7(in_channels=4, hidden_size=1152, injection_layers_map=None):
    """Backward-compatible alias for legacy callsites."""
    return build_adapter_v8(
        in_channels=in_channels,
        hidden_size=hidden_size,
        injection_layers_map=injection_layers_map,
    )


MultiLevelAdapterV7 = MultiLevelAdapterV8
