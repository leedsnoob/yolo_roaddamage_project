import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNv2(nn.Module):
    """Deformable Convolution v2 with BN and activation for RG11 PAN downsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        deformable_groups=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        offset_mask_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            offset_mask_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            self.deformable_groups,
            True,
        )
        return self.act(self.bn(x))

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


class BiFPN_Fuse(nn.Module):
    """BiFPN-style weighted feature fusion for RG11 second-region fusion blocks."""

    def __init__(self, c2, n_inputs=2):
        super().__init__()
        self.c2 = c2
        self.n_inputs = n_inputs
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = 1e-4
        self._align_convs = nn.ModuleDict()

    def forward(self, x):
        assert len(x) == self.n_inputs, f"Expected {self.n_inputs} inputs, got {len(x)}"
        sizes = [xi.shape[2:] for xi in x]
        target_h = min(s[0] for s in sizes)
        target_w = min(s[1] for s in sizes)

        aligned = []
        w = F.relu(self.w)
        w_sum = w.sum() + self.eps
        for i, xi in enumerate(x):
            if xi.shape[2] != target_h or xi.shape[3] != target_w:
                xi = F.adaptive_avg_pool2d(xi, (target_h, target_w))
            if xi.shape[1] != self.c2:
                key = f"align_{i}_{xi.shape[1]}"
                if key not in self._align_convs:
                    conv = nn.Conv2d(xi.shape[1], self.c2, 1, bias=False).to(xi.device)
                    nn.init.kaiming_normal_(conv.weight)
                    self._align_convs[key] = conv
                xi = self._align_convs[key](xi)
            aligned.append(w[i] / w_sum * xi)
        return sum(aligned)


class DACA(nn.Module):
    """Direction-Adaptive Crack Attention used by the LSKA/DySample/ShapeIoU exploration branch."""

    def __init__(self, dim, k_size=11, reduction=4):
        super().__init__()
        pad = k_size // 2
        self.dw_v = nn.Conv2d(dim, dim, (k_size, 1), padding=(pad, 0), groups=dim, bias=False)
        self.bn_v = nn.BatchNorm2d(dim)
        self.dw_h = nn.Conv2d(dim, dim, (1, k_size), padding=(0, pad), groups=dim, bias=False)
        self.bn_h = nn.BatchNorm2d(dim)
        mid = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 1),
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // reduction, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 8), dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        fv = F.silu(self.bn_v(self.dw_v(x)))
        fh = F.silu(self.bn_h(self.dw_h(x)))
        gate_weights = self.gate(torch.cat([fv, fh], dim=1)).softmax(dim=1)
        f_fused = gate_weights[:, 0:1] * fv + gate_weights[:, 1:2] * fh
        return x + self.channel_att(f_fused) * f_fused


class C2_DACA(nn.Module):
    """C2PSA-style wrapper for DACA with the same YAML interface as C2PSA."""

    def __init__(self, c1, c2, n=1, e=0.5, k_size=11):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        from ultralytics.nn.modules.conv import Conv

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(DACA(self.c, k_size) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class StripConvAttention(nn.Module):
    """Strip Convolution Attention for direction-aware feature enhancement.

    Uses horizontal (1×K) and vertical (K×1) depthwise strip convolutions
    to capture directional features of elongated road damage (cracks).
    Inspired by StripRFNet (SRFM) and LSKA.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels (must equal c1).
        k (int): Strip convolution kernel size. Default 7.
    """

    def __init__(self, c1, c2, k=7):
        super().__init__()
        assert c1 == c2, f"StripConvAttention requires c1==c2, got {c1} vs {c2}"
        # Horizontal strip: 1×K depthwise conv
        self.strip_h = nn.Conv2d(c1, c1, kernel_size=(1, k), padding=(0, k // 2), groups=c1, bias=False)
        self.bn_h = nn.BatchNorm2d(c1)
        # Vertical strip: K×1 depthwise conv
        self.strip_v = nn.Conv2d(c1, c1, kernel_size=(k, 1), padding=(k // 2, 0), groups=c1, bias=False)
        self.bn_v = nn.BatchNorm2d(c1)
        # Fusion: 1×1 conv to generate attention map
        self.fusion = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1 // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 // 4, c1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Horizontal and vertical strip convolutions
        h_feat = self.bn_h(self.strip_h(x))
        v_feat = self.bn_v(self.strip_v(x))
        # Combine both directions
        combined = h_feat + v_feat
        # Generate attention weights
        attn = self.fusion(combined)
        return x * attn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class DySample(nn.Module):
    """Dynamic upsampler with optional channel projection.

    This keeps the RG11 top-down path learnable without changing the overall graph logic.
    When out_channels matches in_channels, it behaves like a drop-in upsampler.
    """

    def __init__(self, in_channels, out_channels=None, scale_factor=2, groups=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.groups = groups
        self.offset = nn.Conv2d(in_channels, 2 * groups * scale_factor * scale_factor, 1, bias=False)
        nn.init.trunc_normal_(self.offset.weight, std=0.001)
        self.scope = nn.Conv2d(in_channels, groups * scale_factor * scale_factor, 1, bias=False)
        nn.init.constant_(self.scope.weight, 0.0)
        self.proj = (
            nn.Identity()
            if out_channels is None or out_channels == in_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
        )

    def _generate_grid(self, h, w, device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid_x = 2 * grid_x / max(w - 1, 1) - 1
        grid_y = 2 * grid_y / max(h - 1, 1) - 1
        return grid_x, grid_y

    def forward(self, x):
        b, _, h, w = x.shape
        sf = self.scale_factor

        offset = self.offset(x)
        scope = self.scope(x).sigmoid() * 0.5

        offset = offset.view(b, 2, self.groups, sf, sf, h, w)
        offset = offset.permute(0, 1, 2, 5, 3, 6, 4).reshape(b, 2 * self.groups, h * sf, w * sf)

        scope = scope.view(b, self.groups, sf, sf, h, w)
        scope = scope.permute(0, 1, 4, 2, 5, 3).reshape(b, self.groups, h * sf, w * sf)

        offset_x = offset[:, :self.groups] * scope
        offset_y = offset[:, self.groups:] * scope

        grid_x, grid_y = self._generate_grid(h * sf, w * sf, x.device)
        offset_x = offset_x.mean(dim=1) + grid_x.unsqueeze(0)
        offset_y = offset_y.mean(dim=1) + grid_y.unsqueeze(0)
        grid = torch.stack([offset_x, offset_y], dim=-1)

        x_up = F.interpolate(x, size=(h * sf, w * sf), mode="nearest")
        x_up = F.grid_sample(x_up, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return self.proj(x_up)


class binary_spatial_Attention(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.psi(x)


class DWSConvDown(nn.Module):
    """Depthwise Separable Convolution Downsampling (Drone-YOLO style).

    Large-kernel depthwise conv (stride=2) followed by pointwise conv.
    Used in sandwich-fusion for extracting spatial info from shallower features
    while downsampling to match the target resolution.

    Reference: Drone-YOLO (Zhang, 2023) - sandwich-fusion module, Figure 7.
    """

    def __init__(self, c1, c2, k=7, s=2):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x
