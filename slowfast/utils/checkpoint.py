from slowfast.config import configs

import os
import torch


def save(model, optimizer, epoch):
    checkpoint_path = os.path.join(configs.checkpoint_path, f'checkpoint_{epoch}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, checkpoint_path)


def load(model, optimizer):
    checkpoint_list = [f for f in os.listdir(configs.checkpoint_path) if 'checkpoint' in f]
    if len(checkpoint_list) == 0:
        return 0

    checkpoint_list.sort()
    checkpoint = torch.load(os.path.join(configs.checkpoint_path, checkpoint_list[-1]))
    print(f'Loaded latest checkpoint: {checkpoint_list[-1]}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] + 1
