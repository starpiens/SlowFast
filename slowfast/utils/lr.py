from slowfast.config import configs
import math


def cosine_lr(epoch):
    return configs.cosine_lr_base * 0.5 * (
            math.cos(epoch / configs.max_epoch * math.pi) + 1
    )


def get_lr(epoch):
    if epoch < configs.warmup_epochs:
        # Linear warmup
        start_lr = configs.start_lr
        end_lr = cosine_lr(configs.warmup_epochs)
        delta = (end_lr - start_lr) / configs.warmup_epochs
        lr = delta * epoch + start_lr

    else:
        lr = cosine_lr(epoch)

    return lr
