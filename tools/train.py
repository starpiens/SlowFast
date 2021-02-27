import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.distributed import DistributedSampler

from slowfast.models.slowfast import SlowFast
from slowfast.config import configs
from slowfast.datasets import loader
from slowfast.datasets import utils
import slowfast.utils.distributed as du
from slowfast.utils.meters import TrainMeter, ValMeter
from slowfast.models.build import build_model


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch):
    model.train()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if configs.num_gpus > 0:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
            labels = labels.cuda()

        # TODO: Update the learning rate.
        # lr = 0.0001
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        # Forward pass
        preds = model(inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(preds, labels)
        print(f'epoch: {cur_epoch}, iter: {cur_iter}, loss: {loss}')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_epoch(val_loader, model, val_meter, cur_epoch):
    model.eval()
    pass


def train():
    # Setup environment.
    # du.init_distributed_training()
    np.random.seed(42)
    torch.manual_seed(42)

    # Create model.
    model = build_model('ResNet-18')

    # Construct the optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(),
        0.1,
        0.9,
        weight_decay=0.0001,
        nesterov=True
    )

    # TODO: Load a checkpoint to resume training if applicable.
    start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader('train')
    val_loader = loader.construct_loader('train')

    # Create meters.
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Train.
    for cur_epoch in range(start_epoch, configs.max_epoch):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(cur_epoch)

        print(f"Starting epoch {cur_epoch}")
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch)
        eval_epoch(train_loader, model, val_meter, cur_epoch)


if __name__ == '__main__':
    train()
