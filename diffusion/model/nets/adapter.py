import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.nets.srconvnet_blocks import SRConvNetBlock


class SRConvNetLSAAdapter(nn.Module):
    def __init__(self, hidden_size: int = 1152):
        super().__init__()
        self.hidden_size = int(hidden_size)

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
        self.out_proj = nn.Conv2d(768, self.hidden_size, 1)

        for m in [self.proj2, self.proj3, self.proj4, self.out_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return (1.0 + gamma[:, :, None, None]) * feat + beta[:, :, None, None]

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

        f2_32 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        c2 = self.proj2(f2_32)
        c3 = self.proj3(f3)
        c4 = self.proj4(f4)
        cond_map = self.out_proj(torch.cat([c2, c3, c4], dim=1))
        cond_tokens = cond_map.flatten(2).transpose(1, 2)

        return {
            "cond_tokens": cond_tokens,
            "cond_maps": [f2, f3, f4],
        }


def build_adapter_v8(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)
