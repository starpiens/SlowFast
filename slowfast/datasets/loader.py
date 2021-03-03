import os
import random
import torch
import torch.utils.data
import av
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import decoder
from . import utils
from slowfast.config import configs


def construct_loader(split):
    if split == "train":
        shuffle = True
        drop_last = True
    elif split == "val":
        shuffle = False
        drop_last = False
    else:
        raise NotImplementedError()

    batch_size_per_gpu = configs.train_batch_size // max(1, configs.num_gpus)

    dataset = KineticsDataset(configs.dataset_path, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        drop_last=drop_last
    )
    return loader


class KineticsDataset(torch.utils.data.Dataset):

    _label2num = None

    def __init__(self, path, split):
        """
        .csv file format:
        label,youtube_id,time_start,time_end,split,is_cc
        """
        super(KineticsDataset, self).__init__()
        self.split = split
        self._labels = []
        self._path_videos = []

        path_csv = os.path.join(path, 'classes.csv')
        self._construct_label2num(path_csv)

        path_csv = os.path.join(path, f'{split}.csv')
        with open(path_csv, "r") as f:
            for clip_idx, line in enumerate(f.read().splitlines()):
                if not clip_idx:
                    # First row is title
                    continue
                label_str, yt_id, start, end, _, _ = line.split(',')
                self._labels.append(self._label2num[label_str])
                if '"' in label_str:
                    label_str = label_str[1:-1]
                start = '0' * (6 - len(start)) + start
                end = '0' * (6 - len(end)) + end
                path_to_video = os.path.join(
                    path,
                    f'{split}/{label_str}/{yt_id}_{start}_{end}.mp4'
                )
                self._path_videos.append(path_to_video)

    def __getitem__(self, item):
        if self.split in ['train', 'val']:
            # -1 indicates random sampling.
            jitter_scale_min = 256
            jitter_scale_max = 320
            crop_size = 224
        else:
            raise NotImplementedError()

        num_retries = 10
        for i_try in range(num_retries):
            video_container = None
            # Load video.
            try:
                video_container = av.open(self._path_videos[item])
            except Exception as e:
                print(f"Failed to load video from {self._path_videos[item]} with error {e}")

            # Select a random video if the current video was not able to access.
            if video_container is None:
                print(f"Failed to meta load video idx {item} from {self._path_videos[item]}; trial {i_try}")
                if i_try > num_retries // 4:
                    # Let's try another one
                    item = random.randint(0, len(self._path_videos) - 1)
                continue

            # Decode video.
            frames = decoder.decode(video_container,
                                    configs.input_frames // (configs.alpha * configs.T),
                                    configs.alpha * configs.T)

            # If decoding failed, select another video.
            if frames is None:
                print(f'Failed to decode video idx {item} from {self._path_videos[item]}; trial {i_try}')
                if i_try > num_retries // 4:
                    # Let's try another one
                    item = random.randint(0, len(self._path_videos) - 1)
                continue

            # Color normalization
            frames = utils.tensor_normalize(frames, mean=0.45, std=0.225)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                min_scale=jitter_scale_min,
                max_scale=jitter_scale_max,
                crop_size=crop_size
            )

            label = self._labels[item]
            frames = utils.pack_pathway_output(frames)
            return frames, label, item, {}

        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def __len__(self):
        return len(self._path_videos)

    def _construct_label2num(self, path_csv):
        if self._label2num is not None:
            return
        self._label2num = {}
        with open(path_csv, 'r') as f:
            for idx, label in enumerate(f.read().splitlines()):
                self._label2num[label] = idx
