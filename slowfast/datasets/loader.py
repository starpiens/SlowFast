import os


def load_dataset(path, split):
    """
    .csv file format:
    label youtube_id time_start time_end split
    """
    path_to_file = os.path.join(path, f"{split}.csv")
    labels = []
    path = []
    with open(path_to_file, "r") as f:
        for clip_idx, line in enumerate(f.read().splitlines()):
            if not clip_idx:
                continue
            label, yt_id, start, end, split = line.split(',')
            labels.append(label)
            start = '0' * (6 - len(start)) + start
            end = '0' * (6 - len(end)) + end
            path.append(f"{yt_id}_{start}_{end}.mp4")
    return labels, path
