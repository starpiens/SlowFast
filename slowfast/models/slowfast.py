import torch.nn as nn
import numpy as np
from .resnet import Res18Block, Res50Block, ResStage
from slowfast.config import configs


class Fuse(nn.Module):
    """
    Time-strided convolution fusion.
    """

    def __init__(self):
        super(Fuse, self).__init__()

    def forward(self, x):
        return x


class SlowFast(nn.Module):

    def __init__(self,
                 ):
        super(SlowFast, self).__init__()

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

    def forward(self, x):
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x
