import torch
import torch.nn as nn
from .resnet import Res18Block, Res50Block, ResStage, ResConv, ResHead
from slowfast.config import configs


class Fuse(nn.Module):
    """
    Time-strided convolution fusion.
    """

    def __init__(self,
                 dim_in):
        super(Fuse, self).__init__()
        self.conv = nn.Conv3d(
            dim_in,
            dim_in * 2,
            kernel_size=(5, 1, 1),
            stride=(configs.alpha, 1, 1),
            padding=(2, 0, 0),
            bias=False
        )
        self.bn = nn.BatchNorm3d(dim_in * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_slow = x[0]
        x_fast = x[1]
        fuse = self.conv(x_fast)
        fuse = self.bn(fuse)
        fuse = self.act(fuse)
        x_slow = torch.cat([x_slow, fuse], 1)   # Channel-wise concat
        return x_slow, x_fast


class SlowFast(nn.Module):

    def __init__(self,
                 ):
        super(SlowFast, self).__init__()

        self.stage1 = ResConv()
        self.stage1_fuse = Fuse(configs.dim_out[1] // configs.beta_inv)

        self.stage2 = ResStage(
            dim_in=(configs.dim_out[1] + 2 * configs.dim_out[1] // configs.beta_inv,
                    configs.dim_out[1] // configs.beta_inv),
            dim_inner=(configs.dim_inner[2], configs.dim_inner[2] // configs.beta_inv),
            dim_out=(configs.dim_out[2], configs.dim_out[2] // configs.beta_inv),
            temp_kernel_size=(1, 3),
            stride=1,
            num_blocks=configs.blocks[2],
            block=Res50Block
        )
        self.stage2_fuse = Fuse(configs.dim_out[2] // configs.beta_inv)

        self.stage3 = ResStage(
            dim_in=(configs.dim_out[2] + 2 * configs.dim_out[2] // configs.beta_inv,
                    configs.dim_out[2] // configs.beta_inv),
            dim_inner=(configs.dim_inner[3], configs.dim_inner[3] // configs.beta_inv),
            dim_out=(configs.dim_out[3], configs.dim_out[3] // configs.beta_inv),
            temp_kernel_size=(1, 3),
            stride=2,
            num_blocks=configs.blocks[3],
            block=Res50Block
        )
        self.stage3_fuse = Fuse(configs.dim_out[3] // configs.beta_inv)

        self.stage4 = ResStage(
            dim_in=(configs.dim_out[3] + 2 * configs.dim_out[3] // configs.beta_inv,
                    configs.dim_out[3] // configs.beta_inv),
            dim_inner=(configs.dim_inner[4], configs.dim_inner[4] // configs.beta_inv),
            dim_out=(configs.dim_out[4], configs.dim_out[4] // configs.beta_inv),
            temp_kernel_size=(3, 3),
            stride=2,
            num_blocks=configs.blocks[4],
            block=Res50Block
        )
        self.stage4_fuse = Fuse(configs.dim_out[4] // configs.beta_inv)

        self.stage5 = ResStage(
            dim_in=(configs.dim_out[4] + 2 * configs.dim_out[4] // configs.beta_inv,
                    configs.dim_out[4] // configs.beta_inv),
            dim_inner=(configs.dim_inner[5], configs.dim_inner[5] // configs.beta_inv),
            dim_out=(configs.dim_out[5], configs.dim_out[5] // configs.beta_inv),
            temp_kernel_size=(3, 3),
            stride=2,
            num_blocks=configs.blocks[5],
            block=Res50Block
        )

        self.head = ResHead()

    def forward(self, x):
        """
        x: (x_slow, x_fast)
        """
        x = self.stage1(x)
        x = self.stage1_fuse(x)
        x = self.stage2(x)
        x = self.stage2_fuse(x)
        x = self.stage3(x)
        x = self.stage3_fuse(x)
        x = self.stage4(x)
        x = self.stage4_fuse(x)
        x = self.stage5(x)
        x = self.head(x)
        return x
