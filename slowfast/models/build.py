from .slowfast import SlowFast
from slowfast.config import configs
import torch


def build_model(backbone):
    model = SlowFast(backbone)
    # Move model to GPU if available.
    if configs.num_gpus > 0:
        assert torch.cuda.is_available() is True
        model = model.cuda(device=torch.cuda.current_device())
    return model
