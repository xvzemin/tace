################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


class IndexedSource(SequenceABC):
    """A zero-copy subset retaining exactly the supplied sample order."""

    def __init__(self, source, indices):
        self.source = source
        indices = np.asarray(list(indices))
        if indices.size and (indices.ndim != 1 or indices.dtype.kind not in "iu"):
            raise TypeError(
                "Dataset split indices must be a one-dimensional integer sequence"
            )
        self.indices = indices.astype(np.int64)
        if np.any(self.indices < 0) or np.any(self.indices >= len(source)):
            raise IndexError("Dataset split contains out-of-range indices")

    def __len__(self):
        return len(self.indices)

    def close(self):
        close = getattr(self.source, "close", None)
        if close is not None:
            close()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return IndexedSource(self.source, self.indices[index])
        return self.source[int(self.indices[index])]

    def getitems(self, indices):
        indices = [int(self.indices[index]) for index in indices]
        method = getattr(self.source, "getitems", None)
        return method(indices) if method else [self.source[index] for index in indices]

    __getitems__ = getitems

    def get_records(self, indices):
        from .records import record_from_atoms

        indices = [int(self.indices[index]) for index in indices]
        method = getattr(self.source, "get_records", None)
        return (
            method(indices)
            if method
            else [record_from_atoms(self.source[index]) for index in indices]
        )

    def get_record(self, index):
        return self.get_records([index])[0]

    def __iter__(self):
        for start in range(0, len(self), 128):
            yield from self.getitems(range(start, min(start + 128, len(self))))


def random_split(
    train_list: Sequence, valid_ratio: float, seed: int
) -> Tuple[Sequence, Sequence]:
    """
    The reason why both indices need to be saved is that when I tried to
    reproduce my own results, I found that only loading the valid indices
    would lead to different training results. This happens because the order of
    structures in the training set changes during loading, which in turn alters
    the order of batches.
    """
    size = len(train_list)
    train_size = size - int(valid_ratio * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_index_file = Path(".") / "train.index"  # index starts from zero
    valid_index_file = Path(".") / "valid.index"  # index starts from zero
    with open(train_index_file, "w") as f:
        for idx in indices[:train_size]:  # not allowed sorted
            f.write(f"{idx}\n")
    with open(valid_index_file, "w") as f:
        for idx in indices[train_size:]:
            f.write(f"{idx}\n")

    return (
        IndexedSource(train_list, indices[:train_size]),
        IndexedSource(train_list, indices[train_size:]),
    )
