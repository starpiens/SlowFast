from .slowfast import SlowFast
from slowfast.config import configs
import torch


def build_model(backbone):
    model = SlowFast(backbone)
    if configs.num_gpus > 0:
        model = model.cuda(device=torch.cuda.current_device())
    return model
