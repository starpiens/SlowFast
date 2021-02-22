import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Single residual block.
    """

    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 temp_kernel_size,
                 stride):
        super(ResBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

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

        if hasattr(self, "conv0"):
            x = self.conv0_bn(self.conv0(x))
        x = x + f_x
        x = self.act(x)

        # TODO: Drop connection

        return x


class ResStage(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 temp_kernel_size,
                 stride,
                 num_blocks):
        super(ResStage, self).__init__()
        self.num_blocks = num_blocks

        for pathway in range(2):
            for i in range(num_blocks):
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_inner[pathway],
                    dim_out[pathway],
                    temp_kernel_size,
                    stride if i == 0 else 1
                )
                self.add_module(f'res_{i}_{pathway}', res_block)

    def forward(self, inputs):
        output = []
        for pathway in range(2):
            x = inputs[pathway]
            for i in range(self.num_blocks):
                m = getattr(self, f'res_{i}_{pathway}')
                x = m(x)
            output.append(x)

        return output
