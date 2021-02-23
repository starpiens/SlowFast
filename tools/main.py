import torch
import torch.nn as nn
from slowfast.models.slowfast import SlowFast
from slowfast.config import configs
from slowfast.datasets import loader
from slowfast.datasets import utils
from torch.utils.data.

def train():
    pass


def test():
    pass


def main():
    # Create model
    model = SlowFast("ResNet-18")
    model = nn.parallel.DistributedDataParallel(model)

    train_loader = loader.construct_loader('train')
    val_loader = loader.construct_loader('val')



if __name__ == '__main__':
    main()
