import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import math

from slowfast.models.slowfast import SlowFast
from slowfast.config import configs
from slowfast.datasets import loader
from slowfast.datasets import utils
from slowfast.models.build import build_model
from slowfast.utils.metrics import num_topK_correct
from slowfast.utils import checkpoint as cu


def train_epoch(train_loader, model, optimizer, cur_epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    num_iters = len(train_loader)
    data_size = num_iters * configs.train_batch_size

    sum_loss = 0
    sum_top1_correct = 0
    sum_top5_correct = 0

    with tqdm(total=num_iters, desc=f'Epoch {cur_epoch}') as pbar:
        for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
            # Transfer the data to the current GPU device.
            if configs.num_gpus > 0:
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # TODO: Update the learning rate.
            # lr = 0.0001
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr

            # Forward pass
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            if math.isnan(loss):
                raise RuntimeError("Got NaN loss")
            top1_correct, top5_correct = num_topK_correct(preds, labels, (1, 5))

            # Training stats
            sum_loss += loss
            sum_top1_correct += top1_correct
            sum_top5_correct += top5_correct

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update()

    print(f'loss: {sum_loss / num_iters: .4f}, ',
          f'top1 acc: {sum_top1_correct / data_size * 100: .4f}%, '
          f'top5 acc: {sum_top5_correct / data_size * 100: .4f}%', flush=True)


def eval_epoch(val_loader, model, val_meter, cur_epoch):
    model.eval()
    pass


def train():
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
    start_epoch = cu.load(model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader('train')
    # val_loader = loader.construct_loader('val')

    # Train.
    for epoch in range(start_epoch, configs.max_epoch):
        train_epoch(train_loader, model, optimizer, epoch)
        cu.save(model, optimizer, epoch)
        # eval_epoch(train_loader, model, val_meter, cur_epoch)


if __name__ == '__main__':
    train()
