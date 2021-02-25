# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""

import torch.distributed as dist
from slowfast.config import configs


_LOCAL_PROCESS_GROUP = None


def init_distributed_training():
    """
    Initialize variables needed for distributed training.
    """
    if configs.num_gpus <= 1:
        return
    ranks_on_i = list(range(0, configs.num_gpus))
    pg = dist.new_group(ranks_on_i)
    global _LOCAL_PROCESS_GROUP
    _LOCAL_PROCESS_GROUP = pg
