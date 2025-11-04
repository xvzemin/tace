################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path
from typing import List, Sequence, Tuple


import numpy as np


def random_split(
    train_list: Sequence, valid_ratio: float, seed: int
) -> Tuple[List, List]:
    '''
    The reason why both indices need to be saved is that when I tried to 
    reproduce my own results, I found that only loading the valid indices
    would lead to different training results. This happens because the order of 
    structures in the training set changes during loading, which in turn alters 
    the order of batches.
    '''
    size = len(train_list)
    train_size = size - int(valid_ratio * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_index_file = Path(".") / "train.index"  # index starts from zero
    valid_index_file = Path(".") / "valid.index"  # index starts from zero
    with open(train_index_file, "w") as f:
        for idx in indices[:train_size]: # not allowed sorted
            f.write(f"{idx}\n")
    with open(valid_index_file, "w") as f:
        for idx in indices[train_size:]:
            f.write(f"{idx}\n")

    return (
        [train_list[i] for i in indices[:train_size]],
        [train_list[i] for i in indices[train_size:]],
    )
