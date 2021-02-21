import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Single residual block.
    """

    def __init__(self, dim_in, dim_inner, dim_out):
        super(ResBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        # Dimension of the first block in each stage should be 2x.
        if dim_in != dim_out:
            self.conv0 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(1, 2, 2),
                bias=False
            )
            self.conv0_bn = nn.BatchNorm3d(dim_out)

        self.conv1 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            bias=False
        )
        self.conv1_bn = nn.BatchNorm3d(dim_inner)
        self.conv1_act = nn.ReLU(True)

        self.conv2 = nn.Conv3d(
            dim_inner,
            dim_inner,
            kernel_size=(3, 3, 3),
            bias=False
        )
        self.conv2_bn = nn.BatchNorm3d(dim_inner)
        # TODO: Squeeze-and-Excitation module
        self.conv2_act = nn.ReLU(True)

        self.conv3 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            bias=False
        )
        self.conv3_bn = nn.BatchNorm3d(dim_out)
        self.act = nn.ReLU(True)

    def forward(self, x):
        f_x = self.conv1(x)
        f_x = self.conv1_bn(f_x)
        f_x = self.conv1_act(f_x)
        f_x = self.conv2(f_x)
        f_x = self.conv2_bn(f_x)
        f_x = self.conv2_act(f_x)
        f_x = self.conv3(f_x)
        f_x = self.conv3_bn(f_x)

        if self.dim_in == self.dim_out:
            x = x + f_x
        else:
            x = self.conv0_bn(self.conv0(x)) + f_x
        x = self.act(x)

        # TODO: Drop connection

        return x


class ResStage(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_inner,
                 dim_out,
                 num_blocks):
        super(ResStage, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_blocks = num_blocks

        for i in range(num_blocks):
            res_block = ResBlock(
                dim_in if i == 0 else dim_out,
                dim_inner,
                dim_out,
            )
            self.add_module(f'res_{i}', res_block)

    def forward(self, x):
        for child in self.children():
            x = child(x)
        return x
