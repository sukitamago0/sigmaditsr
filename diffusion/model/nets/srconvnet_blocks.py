import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Vendored/adapted from https://github.com/lifengcs/SRConvNet


class LayerNorm(nn.Module):
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
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class FourierUnit(nn.Module):
    def __init__(self, dim, groups=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted = self.act(self.conv_layer(ffted))
        real2, imag2 = torch.chunk(ffted, 2, dim=1)
        return torch.fft.irfft2(torch.complex(real2, imag2), s=(h, w), dim=(-2, -1), norm="ortho")


class FConvMod(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones(num_heads), requires_grad=True)
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        shortcut = x
        x = self.norm(x)
        a = self.a(x).view(b, self.num_heads, c // self.num_heads, n)
        v = self.v(x).view(b, self.num_heads, c // self.num_heads, n)
        chunk = int(math.ceil(n // 4))
        outs = []
        for a_i, v_i in zip(torch.split(a, chunk, dim=-1), torch.split(v, chunk, dim=-1)):
            outs.append(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * (a_i * v_i))
        x = torch.cat(outs, dim=-1)
        x = F.softmax(x, dim=-1).view(b, c, h, w)
        x = x + self.cpe(shortcut)
        return self.proj(x) + shortcut


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
        weight = torch.mm(attention, self.weight.contiguous().view(self.num_kernels, -1)).contiguous().view(
            b * self.dim, self.dim // self.groups, self.kernel_size, self.kernel_size
        )
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
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.sigmoid(x.view(x.shape[0], -1))


class DynamicKernelAggregation(nn.Module):
    def __init__(self, dim, kernel_size, groups=1, num_kernels=4):
        super().__init__()
        self.attention = KernelAttention(dim, num_kernels=num_kernels)
        self.aggregation = KernelAggregation(dim, kernel_size=kernel_size, groups=groups, num_kernels=num_kernels)

    def forward(self, x):
        return self.aggregation(x, self.attention(x))


class DyConv(nn.Module):
    def __init__(self, dim, kernel_size, groups, num_kernels=1):
        super().__init__()
        self.conv = DynamicKernelAggregation(dim, kernel_size=kernel_size, groups=groups, num_kernels=num_kernels) if num_kernels > 1 else nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=groups, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)


class MixFFN(nn.Module):
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
        b, c, g, h, w = torch.cat([x1, x2], dim=2).shape
        x = torch.cat([x1, x2], dim=2).view(b, c * g, h, w)
        return self.proj_out(x) + shortcut


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
