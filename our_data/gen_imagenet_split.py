import numpy as np
from pathlib import Path


def class_map(path):
    all_clsses = []
    for i in range(1, 10):
        curr_file = open(Path(path) / "index_list" / f"session_{i}.txt", 'r')
        lines = [l for l in curr_file]
        curr_lbls = list(set([l.split('/')[-2] for l in lines]))
        all_clsses += curr_lbls

    return {k: v for v, k in enumerate(all_clsses)}


dataroot = Path('/mnt/dsi_vol1/users/aviv_navon/mini-imagenet/data')
prefix = 'images'
outpath = Path('/mnt/dsi_vol1/users/aviv_navon/mini-imagenet/data/pod_index')
label_map = class_map(dataroot)

# train
all_train_images = []
order = []

gen_sample = False

if gen_sample:
    for episode_id in range(1, 10):
        idx_file = open(dataroot / "index_list" / f"session_{episode_id}.txt", 'r')
        lines = [l for l in idx_file]

        idxs = np.random.choice(range(len(lines)), 60 * 10)

        order += list(set([label_map[l.split("/")[-2]] for i, l in enumerate(lines) if i in idxs]))
        all_train_images += [f'{prefix}/{Path(l).name.rstrip()} {label_map[l.split("/")[-2]]}' for i, l in enumerate(lines) if i in idxs]

    with open(outpath / 'train_100.txt', 'w') as f:
        for item in all_train_images:
            f.write(f'{item}\n')

    # test
    all_test_images = []
    order = []

    for episode_id in range(1, 10):
        idx_file = open(dataroot / "index_list" / f"test_{episode_id}.txt", 'r')
        lines = [l for l in idx_file]
        curr_classes = list(set([label_map[l.split("/")[-2]] for l in lines]))
        order += curr_classes
        all_test_images += [f'{prefix}/{Path(l).name.rstrip()} {label_map[l.split("/")[-2]]}' for l in lines]

    with open(outpath / 'val_100.txt', 'w') as f:
        for item in all_test_images:
            f.write(f'{item}\n')

else:
    for episode_id in range(1, 10):
        if episode_id == 1:
            idx_file = open(dataroot / "index_list" / f"session_{episode_id}_train.txt", 'r')
        else:
            idx_file = open(dataroot / "index_list" / f"session_{episode_id}.txt", 'r')
        lines = [l for l in idx_file]
        order += list(set([label_map[l.split("/")[-2]] for l in lines]))
        all_train_images += [f'{prefix}/{Path(l).name.rstrip()} {label_map[l.split("/")[-2]]}' for l in lines]

    with open(outpath / 'train_100.txt', 'w') as f:
        for item in all_train_images:
            f.write(f'{item}\n')

    # test
    all_test_images = []
    order = []

    for episode_id in range(1, 10):
        idx_file = open(dataroot / "index_list" / f"test_{episode_id}.txt", 'r')
        lines = [l for l in idx_file]
        curr_classes = list(set([label_map[l.split("/")[-2]] for l in lines]))
        order += curr_classes
        all_test_images += [f'{prefix}/{Path(l).name.rstrip()} {label_map[l.split("/")[-2]]}' for l in lines]

    with open(outpath / 'val_100.txt', 'w') as f:
        for item in all_test_images:
            f.write(f'{item}\n')
