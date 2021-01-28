import numpy as np
from pathlib import Path

dataroot = Path('/Users/avivnavon/Desktop/inc-gp/cub')
outpath = Path('./cub')
# dataroot = Path('/cortex/users/idan/projects/IGP/incremental_gp/experiments/IL/cub')


def get_class_map(path):
    path = Path(path)
    curr_file = open(path / "index_list" / f"train.txt", 'r')
    lines = [l for l in curr_file]
    id_cls = set([tuple(l.split('/')[2].split('.')) for l in lines])
    return {v[1]: int(v[0]) - 1 for v in id_cls}


cls_mapping = get_class_map(dataroot)

# train
train_file = open(dataroot / "index_list" / f"train.txt", 'r')
train_img = ['/'.join(l.split('/')[2:]).rstrip() for l in train_file]

# val
val_file = open(dataroot / f"val_indices", 'r')
val_img = [l.split(' ')[1].rstrip() for l in val_file]

# test
test_file = open(dataroot / "index_list" / f"test.txt", 'r')
test_img = ['/'.join(l.split('/')[2:]).rstrip() for l in test_file]

print(f"train {len(train_img)}, test {len(test_img)}, val {len(val_img)}")

# filter val from train
train_img = [t for t in train_img if t not in val_img]

print(f"train {len(train_img)}, test {len(test_img)}, val {len(val_img)}")

with open(outpath / 'train.txt', 'w') as f:
    for item in train_img:
        f.write(f'{item} {int(item.split(".")[0]) - 1}\n')

with open(outpath / 'val.txt', 'w') as f:
    for item in val_img:
        f.write(f'{item} {int(item.split(".")[0]) - 1}\n')

with open(outpath / 'test.txt', 'w') as f:
    for item in test_img:
        f.write(f'{item} {int(item.split(".")[0]) - 1}\n')
