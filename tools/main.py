import torch
from slowfast.models.slowfast import SlowFast
from slowfast.config import configs

def train():
    pass


def test():
    pass


def main():
    x_slow = torch.rand((10, 3, 4, 224, 224))
    x_fast = torch.rand((10, 3, 32, 224, 224))
    model = SlowFast(configs.backbone)
    y = model.forward([x_slow, x_fast])
    print(y)
    print(y.shape)
    print(model)


if __name__ == '__main__':
    main()
