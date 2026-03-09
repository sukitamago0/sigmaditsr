import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Vendored and adapted from SRConvNet (https://github.com/lifengcs/SRConvNet)
# Core idea preserved: FConvMod (FMA-like) + MixFFN (DML-like) in each block.


class LayerNorm(nn.Module):
    """From SRConvNet ConvNeXt-style LayerNorm (channels_first path)."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.data_format != "channels_first":
            raise NotImplementedError
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FourierUnit(nn.Module):
    """SRConvNet FourierUnit, updated to torch.fft API for compatibility."""

    def __init__(self, dim, groups=1):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        real = ffted.real
        imag = ffted.imag
        ffted = torch.cat([real, imag], dim=1)
        ffted = self.conv_layer(ffted)
        ffted = self.act(ffted)
        real2, imag2 = torch.chunk(ffted, 2, dim=1)
        comp = torch.complex(real2, imag2)
        out = torch.fft.irfft2(comp, s=(h, w), dim=(-2, -1), norm="ortho")
        return out


class FConvMod(nn.Module):
    """SRConvNet frequency-spatial modulation aggregation (FMA)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        shortcut = x
        pos_embed = self.cpe(x)
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)

        head_c = c // self.num_heads
        a = a.view(b, self.num_heads, head_c, n)
        v = v.view(b, self.num_heads, head_c, n)

        chunk = int(math.ceil(n // 4))
        a_all = torch.split(a, chunk, dim=-1)
        v_all = torch.split(v, chunk, dim=-1)
        attns = []
        for a_i, v_i in zip(a_all, v_all):
            attn = a_i * v_i
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)

        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)
        x = x.view(b, c, h, w)
        x = x + pos_embed
        x = self.proj(x)
        return x + shortcut


class KernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels, bias=True):
        super().__init__()
        self.groups = groups
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(num_kernels, dim, dim // groups, kernel_size, kernel_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_kernels, dim)) if bias else None
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, attention):
        b, c, h, w = x.shape
        x = x.contiguous().view(1, b * self.dim, h, w)
        weight = self.weight.contiguous().view(self.num_kernels, -1)
        weight = torch.mm(attention, weight).contiguous().view(b * self.dim, self.dim // self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            bias = torch.mm(attention, self.bias).contiguous().view(-1)
            x = F.conv2d(x, weight=weight, bias=bias, stride=1, padding=self.kernel_size // 2, groups=self.groups * b)
        else:
            x = F.conv2d(x, weight=weight, bias=None, stride=1, padding=self.kernel_size // 2, groups=self.groups * b)
        return x.contiguous().view(b, self.dim, x.shape[-2], x.shape[-1])


class KernelAttention(nn.Module):
    def __init__(self, dim, reduction=8, num_kernels=8):
        super().__init__()
        mid_channels = dim // reduction if dim != 3 else num_kernels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, mid_channels, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, num_kernels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        return self.sigmoid(x)


class DynamicKernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups=1, num_kernels=4):
        super().__init__()
        assert dim % groups == 0
        self.attention = KernelAttention(dim, num_kernels=num_kernels)
        self.aggregation = KernelAggregation(dim, kernel_size=kernel_size, groups=groups, num_kernels=num_kernels)

    def forward(self, x):
        attention = self.attention(x)
        return self.aggregation(x, attention)


class DyConv(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels=1):
        super().__init__()
        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(dim, kernel_size=kernel_size, groups=groups, num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=groups, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)


class MixFFN(nn.Module):
    """SRConvNet multi-scale dynamic local modeling (DML)."""

    def __init__(self, dim, num_kernels):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.conv1 = DyConv(dim, kernel_size=5, groups=dim, num_kernels=num_kernels)
        self.conv2 = DyConv(dim, kernel_size=7, groups=dim, num_kernels=num_kernels)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.act(self.proj_in(x))
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.act(self.conv1(x1)).unsqueeze(dim=2)
        x2 = self.act(self.conv2(x2)).unsqueeze(dim=2)
        x = torch.cat([x1, x2], dim=2)
        b, c, g, h, w = x.shape
        x = x.view(b, c * g, h, w)
        x = self.proj_out(x)
        return x + shortcut


class FMA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.core = FConvMod(dim, num_heads=num_heads)

    def forward(self, x):
        return self.core(x)


class DML(nn.Module):
    def __init__(self, dim, num_kernels=4):
        super().__init__()
        self.core = MixFFN(dim, num_kernels=num_kernels)

    def forward(self, x):
        return self.core(x)


class SRConvNetBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, num_kernels: int = 4):
        super().__init__()
        self.fma = FMA(channels, num_heads=num_heads)
        self.dml = DML(channels, num_kernels=num_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fma(x)
        x = self.dml(x)
        return x


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
        fused = torch.cat([c2, c3, c4], dim=1)
        cond_map = self.out_proj(fused)
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
