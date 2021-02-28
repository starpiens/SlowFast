import math
import torch.nn.functional as F
import numpy as np


def random_short_side_scale_jitter(images, min_size, max_size):
    new_size = int(round(np.random.uniform(min_size, max_size)))
    height = images.shape[2]
    width = images.shape[3]
    if width < height:
        new_width = new_size
        new_height = int(height * (float(new_width) / width))
    else:
        new_height = new_size
        new_width = int(width * (float(new_height) / height))

    resized_images = F.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized_images


def random_crop(images, size):
    width = images.shape[3]
    width_offset = np.random.randint(0, width - size) if width > size else 0
    height = images.shape[2]
    height_offset = np.random.randint(0, height - size) if height > size else 0

    cropped_images = images[:, :, height_offset: height_offset + size, width_offset: width_offset + size]
    return cropped_images


def horizontal_flip(prob, images):
    if np.random.uniform() < prob:
        images = images.flip((-1))  # image: C T H W
    return images
