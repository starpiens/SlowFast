import torch
import torch.nn as nn


class X3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm = nn.BatchNorm3d

        # Parameters for X3D XS.
        self.gamma_w = 2.0
        self.gamma_d = 2.2
        self.gamma_b = 2.25

        self.stages = [
            # [blocks, channels]
            [3, 12],
            [5, 24],
            [7, 48],
            [11, 96],
        ]

        num_groups = 1
        width_per_group = 64
        dim_inner = num_groups * width_per_group
