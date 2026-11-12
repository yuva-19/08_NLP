
import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Convolution Block
# --------------------------------------------------

class Conv_block(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride, padding, groups=1):

        super(Conv_block, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x


# --------------------------------------------------
# 2. Linear Block
# --------------------------------------------------

class Linear_block(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride, padding, groups=1):

        super(Linear_block, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        return x


# --------------------------------------------------
# 3. Bottleneck Block
# --------------------------------------------------

class Bottleneck(nn.Module):

    def __init__(self, in_c, out_c, expansion, stride):

        super(Bottleneck, self).__init__()

        mid_c = in_c * expansion

        self.use_residual = (stride == 1 and in_c == out_c)

        self.expand = Conv_block(
            in_c=in_c,
            out_c=mid_c,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.depthwise = Conv_block(
            in_c=mid_c,
            out_c=mid_c,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_c
        )

        self.project = Linear_block(
            in_c=mid_c,
            out_c=out_c,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):

        identity = x

        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.use_residual:
            out = out + identity

        return out


# --------------------------------------------------
# 4. Stage Creation
# --------------------------------------------------

def _make_stage(in_c, out_c, expansion, num_blocks, stride):

    layers = []

    for i in range(num_blocks):

        block_stride = stride if i == 0 else 1
        block_in_c = in_c if i == 0 else out_c

        layers.append(
            Bottleneck(
                in_c=block_in_c,
                out_c=out_c,
                expansion=expansion,
                stride=block_stride
            )
        )

    return nn.Sequential(*layers)


# --------------------------------------------------
# 5. MobileFaceNet Architecture
# --------------------------------------------------

class MobileFaceNet(nn.Module):

    def __init__(self, embedding_size=128):

        super(MobileFaceNet, self).__init__()

        # Initial convolution
        self.conv1 = Conv_block(
            in_c=3,
            out_c=64,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Initial depthwise convolution
        self.conv2 = Conv_block(
            in_c=64,
            out_c=64,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=64
        )

        # Bottleneck stages
        self.stage1 = _make_stage(
            in_c=64,
            out_c=64,
            expansion=2,
            num_blocks=5,
            stride=2
        )

        self.stage2 = _make_stage(
            in_c=64,
            out_c=128,
            expansion=4,
            num_blocks=1,
            stride=2
        )

        self.stage3 = _make_stage(
            in_c=128,
            out_c=128,
            expansion=2,
            num_blocks=6,
            stride=1
        )

        self.stage4 = _make_stage(
            in_c=128,
            out_c=128,
            expansion=4,
            num_blocks=1,
            stride=2
        )

        self.stage5 = _make_stage(
            in_c=128,
            out_c=128,
            expansion=2,
            num_blocks=2,
            stride=1
        )

        # Final convolution
        self.conv6 = Conv_block(
            in_c=128,
            out_c=512,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Final depthwise convolution
        self.conv7 = Conv_block(
            in_c=512,
            out_c=512,
            kernel_size=7,
            stride=1,
            padding=0,
            groups=512
        )

        # Final linear convolution
        self.conv8 = Linear_block(
            in_c=512,
            out_c=embedding_size,
            kernel_size=1,
            stride=1,
            padding=0
        )

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

        x = torch.flatten(x, start_dim=1)

        return x
