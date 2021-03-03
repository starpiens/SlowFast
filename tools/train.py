import torch
import torch.nn as nn

from tqdm import tqdm
import math

from slowfast.config import configs
from slowfast.datasets import loader
from slowfast.models.build import build_model
from slowfast.utils.metrics import num_topK_correct
from slowfast.utils import checkpoint as cu
from slowfast.utils import lr


def train_epoch(loader, model, optimizer, epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    num_iters = len(loader)
    data_size = num_iters * configs.train_batch_size

    sum_loss = 0
    sum_top1_correct = 0
    sum_top5_correct = 0

    with tqdm(total=num_iters, desc=f'Epoch {epoch}, training') as pbar:
        for it, (inputs, labels, _, meta) in enumerate(loader):
            # Transfer the data to the current GPU device.
            if configs.num_gpus > 0:
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            learning_rate = lr.get_lr(epoch + it / num_iters)
            optimizer.param_groups[0]['lr'] = learning_rate

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

    print(f'loss: {sum_loss / num_iters: .4f}, '
          f'top1 acc: {sum_top1_correct / data_size * 100: .4f}%, '
          f'top5 acc: {sum_top5_correct / data_size * 100: .4f}%', flush=True)


def eval_epoch(loader, model, epoch):
    model.eval()
    num_iters = len(loader)

    data_size = 0
    sum_top1_correct = 0
    sum_top5_correct = 0

    with tqdm(total=num_iters, desc=f'Epoch {epoch}, evaluating') as pbar:
        for iter, (inputs, labels, _, meta) in enumerate(loader):
            # Transfer the data to the current GPU device.
            if configs.num_gpus > 0:
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            preds = model(inputs)
            top1_correct, top5_correct = num_topK_correct(preds, labels, (1, 5))

            # Evaluating stats
            sum_top1_correct += top1_correct
            sum_top5_correct += top5_correct
            data_size += len(labels)

            pbar.update()

    print(f'top1 acc: {sum_top1_correct / data_size * 100: .4f}%, '
          f'top5 acc: {sum_top5_correct / data_size * 100: .4f}%', flush=True)


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

    start_epoch = cu.load(model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader('train')
    val_loader = loader.construct_loader('val')

    # Train.
    for epoch in range(start_epoch, configs.max_epoch):
        train_epoch(train_loader, model, optimizer, epoch)
        eval_epoch(val_loader, model, epoch)
        cu.save(model, optimizer, epoch)


if __name__ == '__main__':
    train()
