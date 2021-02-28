import torch
from . import transform
from slowfast.config import configs


def tensor_normalize(tensor, mean, std):
    tensor = tensor.float() / 255.0
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def spatial_sampling(frames, min_scale, max_scale, crop_size):
    frames = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
    frames = transform.random_crop(frames, crop_size)
    frames = transform.horizontal_flip(0.5, frames)
    return frames


def pack_pathway_output(frames):
    fast_input = frames
    slow_input = torch.index_select(
        frames,
        1,
        torch.linspace(
            0,
            frames.shape[1] - 1,
            frames.shape[1] // configs.alpha
        ).long()
    )
    frame_list = [slow_input, fast_input]
    return frame_list
