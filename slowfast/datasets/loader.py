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

    batch_size = int(configs.train_batch_size / max(1, configs.num_gpus))

    dataset = KineticsDataset(configs.dataset_path, split)
    sampler = DistributedSampler(dataset) if configs.num_gpus > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader


class KineticsDataset(torch.utils.data.Dataset):

    _label2num = None

    def __init__(self, path, split):
        """
        .csv file format:
        label youtube_id time_start time_end split
        """
        super(KineticsDataset, self).__init__()
        self.split = split
        self.labels = []
        self.path_videos = []
        self.video_meta = {}

        path_csv = os.path.join(path, 'labels.csv')
        self._construct_label2num(path_csv)

        path_csv = os.path.join(path, 'tmp.csv')
        # self.path_to_file = os.path.join(path, f'{split}.csv')
        with open(path_csv, "r") as f:
            for clip_idx, line in enumerate(f.read().splitlines()):
                if not clip_idx:
                    # First row is title
                    continue
                label_str, yt_id, start, end, split = line.split(',')
                start = '0' * (6 - len(start)) + start
                end = '0' * (6 - len(end)) + end
                path_to_video = os.path.join(
                    path,
                    f'{split}/{label_str}/{yt_id}_{start}_{end}.mp4'
                )
                self.path_videos.append(path_to_video)
                self.labels.append(self._label2num[label_str])
                self.video_meta[clip_idx] = {}

    def __getitem__(self, item):
        if self.split in ['train', 'val']:
            # -1 indicates random sampling.
            temporal_sample_idx = -1
            spatial_sample_idx = -1
            min_scale = 256
            max_scale = 320
            crop_size = 224

        else:
            raise NotImplementedError()

        num_retries = 10

        for i_try in range(num_retries):
            video_container = None
            # Load video.
            try:
                video_container = av.open(self.path_videos[item])
            except Exception as e:
                print(f"Failed to load video from {self.path_videos[item]} with error {e}")

            # Select a random video if the current video was not able to access.
            if video_container is None:
                print(f"Failed to meta load video idx {item} from {self.path_videos[item]}; trial {i_try}")
                if i_try > num_retries // 2:
                    # Let's try another one
                    item = random.randint(0, len(self.path_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames, fps, decode_all_video = decoder.decode(
                video_container,
                2,
                32,
                temporal_sample_idx,
                10,
                self.video_meta[item],
                30,
                'pyav',
                min_scale
            )

            # If decoding failed, select another video.
            if frames is None:
                print(f'Failed to decode video idx {item} from {self.path_videos[item]}; trial {i_try}')
                if i_try > num_retries // 2:
                    # Let's try another one
                    item = random.randint(0, len(self.path_videos) - 1)
                continue

            # Color normalization
            frames = utils.tensor_normalize(frames, [0.45] * 3, [0.225] * 3)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_idx,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=True,
                inverse_uniform_sampling=False
            )

            label = self.labels[item]
            frames = utils.pack_pathway_output(frames)
            return frames, label, item, {}

        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def __len__(self):
        return len(self.path_videos)

    def _construct_label2num(self, path_csv):
        if self._label2num is not None:
            return
        self._label2num = {}
        with open(path_csv, 'r') as f:
            for idx, label in enumerate(f.read().splitlines()):
                self._label2num[label] = idx
