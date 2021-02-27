import torch
import torch.nn as nn
from slowfast.config import configs


class ResConv(nn.Module):

    def __init__(self):
        super(ResConv, self).__init__()

        self.slow_conv = nn.Conv3d(
            3,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False
        )
        self.slow_bn = nn.BatchNorm3d(num_features=64)
        self.slow_act = nn.ReLU(inplace=True)
        self.slow_pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

        self.fast_conv = nn.Conv3d(
            3,
            64 // configs.beta_inv,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False
        )
        self.fast_bn = nn.BatchNorm3d(num_features=64 // configs.beta_inv)
        self.fast_act = nn.ReLU(inplace=True)
        self.fast_pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )

    def forward(self, x):
        x[0] = self.slow_conv(x[0])
        x[0] = self.slow_bn(x[0])
        x[0] = self.slow_act(x[0])
        x[0] = self.slow_pool(x[0])

        x[1] = self.fast_conv(x[1])
        x[1] = self.fast_bn(x[1])
        x[1] = self.fast_act(x[1])
        x[1] = self.fast_pool(x[1])

        return x


class Res18Block(nn.Module):
    """
    Single residual block of ResNet-18.
    """

    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 temp_kernel_size,
                 stride):
        super(Res18Block, self).__init__()

        # Number of channels and spatial size of input should be matched with output
        # because of residual connection.
        if (dim_in != dim_out) or (stride > 1):
            self.conv0 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(1, stride, stride),
                bias=False
            )
            self.conv0_bn = nn.BatchNorm3d(dim_out)

        # Tx3x3, dim_inner.
        self.conv1 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(temp_kernel_size, 3, 3),
            padding=(temp_kernel_size // 2, 1, 1),
            bias=False
        )
        self.conv1_bn = nn.BatchNorm3d(dim_inner)
        self.conv1_act = nn.ReLU(inplace=True)

        # 1x3x3, dim_out. Stride is applied here.
        self.conv2 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
        )
        self.conv2_bn = nn.BatchNorm3d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        f_x = self.conv1(x)
        f_x = self.conv1_bn(f_x)
        f_x = self.conv1_act(f_x)
        f_x = self.conv2(f_x)
        f_x = self.conv2_bn(f_x)

        if hasattr(self, 'conv0'):
            x = self.conv0_bn(self.conv0(x))
        x = x + f_x
        x = self.act(x)

        return x


class Res50Block(nn.Module):
    """
    Single residual block of ResNet-50.
    """

    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 temp_kernel_size,
                 stride):
        super(Res50Block, self).__init__()

        # Number of channels and spatial size of input should be matched with output
        # because of residual connection.
        if (dim_in != dim_out) or (stride > 1):
            self.conv0 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(1, stride, stride),
                bias=False
            )
            self.conv0_bn = nn.BatchNorm3d(dim_out)

        # Tx1x1, dim_inner.
        self.conv1 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(temp_kernel_size, 1, 1),
            padding=(temp_kernel_size // 2, 0, 0),
            bias=False
        )
        self.conv1_bn = nn.BatchNorm3d(dim_inner)
        self.conv1_act = nn.ReLU(inplace=True)

        # 1x3x3, dim_inner. Stride is applied here.
        self.conv2 = nn.Conv3d(
            dim_inner,
            dim_inner,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False
        )
        self.conv2_bn = nn.BatchNorm3d(dim_inner)
        self.conv2_act = nn.ReLU(inplace=True)

        # 1x1x1, dim_out.
        self.conv3 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            bias=False
        )
        self.conv3_bn = nn.BatchNorm3d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        f_x = self.conv1(x)
        f_x = self.conv1_bn(f_x)
        f_x = self.conv1_act(f_x)
        f_x = self.conv2(f_x)
        f_x = self.conv2_bn(f_x)
        f_x = self.conv2_act(f_x)
        f_x = self.conv3(f_x)
        f_x = self.conv3_bn(f_x)

        if hasattr(self, 'conv0'):
            x = self.conv0_bn(self.conv0(x))
        x = x + f_x
        x = self.act(x)

        return x


class ResStage(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 temp_kernel_size,
                 stride,
                 num_blocks,
                 block):
        super(ResStage, self).__init__()
        self.num_blocks = num_blocks

        for pathway in range(2):
            for i in range(num_blocks):
                res_block = block(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_inner[pathway],
                    dim_out[pathway],
                    temp_kernel_size[pathway],
                    stride if i == 0 else 1
                )
                self.add_module('res_' + ('fast' if pathway else 'slow') + f'_{i}', res_block)

    def forward(self, x):
        for pathway in range(2):
            for i in range(self.num_blocks):
                m = getattr(self, 'res_' + ('fast' if pathway else 'slow') + f'_{i}')
                x[pathway] = m(x[pathway])

        return x


class ResHead(nn.Module):

    def __init__(self, dropout_rate=0.5):
        super(ResHead, self).__init__()
        self.slow_pool = nn.AvgPool3d((configs.T, 7, 7))
        self.fast_pool = nn.AvgPool3d((configs.alpha * configs.T, 7, 7))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(configs.dim_out[-1] + configs.dim_out[-1] // configs.beta_inv,
                                configs.num_classes)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x[0] = self.slow_pool(x[0])
        x[0] = x[0].view(x[0].shape[0], -1)
        x[1] = self.fast_pool(x[1])
        x[1] = x[1].view(x[1].shape[0], -1)
        x = torch.cat([x[0], x[1]], 1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        x = self.act(x)
        return x
