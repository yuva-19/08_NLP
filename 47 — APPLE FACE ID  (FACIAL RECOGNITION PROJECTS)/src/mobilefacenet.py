
import torch
import torch.nn as nn


# ============================================================
# 1. CONVOLUTION BLOCK
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x


# ============================================================
# 2. LINEAR BLOCK
# ============================================================

class LinearBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        return x


# ============================================================
# 3. BOTTLENECK BLOCK
# ============================================================

class Bottleneck(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion,
        stride
    ):
        super().__init__()

        hidden_channels = in_channels * expansion

        # Expansion: 1x1 convolution
        self.expand = ConvBlock(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Depthwise: 3x3 convolution
        self.depthwise = ConvBlock(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_channels
        )

        # Projection: 1x1 linear convolution
        self.project = LinearBlock(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Residual connection
        self.use_residual = (
            stride == 1 and in_channels == out_channels
        )

    def forward(self, x):

        identity = x

        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)

        if self.use_residual:
            x = x + identity

        return x


# ============================================================
# 4. MOBILEFACENET
# ============================================================

class MobileFaceNet(nn.Module):

    def __init__(self, embedding_size=128):
        super().__init__()

        # ----------------------------------------------------
        # Initial Convolution Layers
        # ----------------------------------------------------

        self.conv1 = ConvBlock(
            3, 64,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv2 = ConvBlock(
            64, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=64
        )

        # ----------------------------------------------------
        # Bottleneck Stage 1
        # 64 -> 64, Expansion = 2, Blocks = 5
        # ----------------------------------------------------

        self.stage1 = self._make_stage(
            in_channels=64,
            out_channels=64,
            expansion=2,
            num_blocks=5,
            stride=2
        )

        # ----------------------------------------------------
        # Bottleneck Stage 2
        # 64 -> 128, Expansion = 4, Blocks = 1
        # ----------------------------------------------------

        self.stage2 = self._make_stage(
            in_channels=64,
            out_channels=128,
            expansion=4,
            num_blocks=1,
            stride=2
        )

        # ----------------------------------------------------
        # Bottleneck Stage 3
        # 128 -> 128, Expansion = 2, Blocks = 6
        # ----------------------------------------------------

        self.stage3 = self._make_stage(
            in_channels=128,
            out_channels=128,
            expansion=2,
            num_blocks=6,
            stride=1
        )

        # ----------------------------------------------------
        # Bottleneck Stage 4
        # 128 -> 128, Expansion = 4, Blocks = 1
        # ----------------------------------------------------

        self.stage4 = self._make_stage(
            in_channels=128,
            out_channels=128,
            expansion=4,
            num_blocks=1,
            stride=2
        )

        # ----------------------------------------------------
        # Bottleneck Stage 5
        # 128 -> 128, Expansion = 2, Blocks = 2
        # ----------------------------------------------------

        self.stage5 = self._make_stage(
            in_channels=128,
            out_channels=128,
            expansion=2,
            num_blocks=2,
            stride=1
        )

        # ----------------------------------------------------
        # Final Convolution
        # ----------------------------------------------------

        self.conv6 = ConvBlock(
            128, 512,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Global Depthwise Convolution
        self.conv7 = ConvBlock(
            512, 512,
            kernel_size=7,
            stride=1,
            padding=0,
            groups=512
        )

        # Linear Embedding Projection
        self.conv8 = LinearBlock(
            512, embedding_size,
            kernel_size=1,
            stride=1,
            padding=0
        )

    # --------------------------------------------------------
    # Bottleneck Stage Generator
    # --------------------------------------------------------

    def _make_stage(
        self,
        in_channels,
        out_channels,
        expansion,
        num_blocks,
        stride
    ):

        layers = []

        # First block performs the configured stride
        layers.append(
            Bottleneck(
                in_channels,
                out_channels,
                expansion,
                stride
            )
        )

        # Remaining blocks use stride = 1
        for _ in range(1, num_blocks):

            layers.append(
                Bottleneck(
                    out_channels,
                    out_channels,
                    expansion,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    # --------------------------------------------------------
    # Forward Propagation
    # --------------------------------------------------------

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = x.flatten(1)

        return x
