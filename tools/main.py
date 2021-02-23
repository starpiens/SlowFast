import torch
from slowfast.models.slowfast import SlowFast
from slowfast.config import configs
from slowfast.datasets.loader import load_dataset

def train():
    pass


def test():
    pass


def main():
    train_set = load_dataset(configs.dataset_path, "train")
    test_set = load_dataset(configs.dataset_path, "test")
    print(test_set[0])


if __name__ == '__main__':
    main()
