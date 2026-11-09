################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import heapq
from numbers import Integral, Real

import numpy as np
import torch.distributed as distributed
from torch.utils.data import DistributedSampler


def validate_integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class EpochSampler(DistributedSampler):
    """Sample consecutive parts of a repeating, seeded dataset permutation.

    Parameters
    ----------
    dataset : Sized
        Nonempty training dataset. Its length and ordering must remain fixed.
    epoch_size : int or float
        Global samples per epoch, or a fraction in ``(0, 1]`` of the dataset.
        Fractions are rounded down, with a minimum of one sample. The resolved
        count must not exceed the dataset size. Actual epoch size is rounded up
        to equal rank lengths, or full global batches when ``drop_last=True``.
    num_replicas, rank : int, optional
        Distributed world size and rank. Default to the initialized process
        group, or one replica and rank zero outside distributed training.
    shuffle : bool
        Apply a seeded permutation; otherwise traverse dataset indices in order.
    seed : int
        Seed shared by every rank. The permutation stays fixed across epochs.
    batch_size : int
        Per-rank batch size.
    drop_last : bool
        Require full batches. Additional samples are taken from the continuing
        stream, including the next dataset pass when necessary.
    balance : {"atoms", "edges"}, pair of int, or None
        Redistribute each global batch using cached graph sizes. Preserve its
        samples and equal per-rank counts. Requires ``dataset.get_sizes()``.
        A pair gives nonnegative atom and edge weights, respectively.

    Notes
    -----
    Every consecutive ``len(dataset)`` positions in the global stream visit
    every sample once, even when an epoch crosses a dataset-pass boundary.
    Ranks receive strided portions of the global stream. No sample repeats
    before a complete dataset pass. A seeded
    Feistel permutation with cycle walking avoids storing a dataset-sized
    permutation.

    Iteration never commits progress: call :meth:`advance` only for samples in
    completed training batches. This separates prefetch from checkpoint state.
    ``len(sampler)`` always describes a full local epoch, including after resume.

    Examples
    --------
    >>> sampler = EpochSampler(range(10), epoch_size=4, shuffle=False)
    >>> list(sampler)
    [0, 1, 2, 3]
    >>> sampler.advance(2)
    >>> list(sampler)
    [2, 3]
    >>> sampler.set_epoch(1)
    >>> list(sampler)
    [2, 3, 4, 5]
    """

    _chunk_size = 4096

    def __init__(
        self,
        dataset,
        epoch_size=1.0,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        batch_size=1,
        drop_last=False,
        balance=None,
    ):
        size = len(dataset)
        if not size:
            raise ValueError("EpochSampler requires a nonempty dataset")
        if isinstance(epoch_size, bool):
            raise TypeError("epoch_size must be an integer count or float fraction")
        if isinstance(epoch_size, Integral):
            count = int(epoch_size)
        elif isinstance(epoch_size, Real):
            fraction = float(epoch_size)
            if not 0 < fraction <= 1:
                raise ValueError("epoch_size fraction must be in (0, 1]")
            count = size if fraction == 1.0 else max(1, min(size, int(size * fraction)))
        else:
            raise TypeError("epoch_size must be an integer count or float fraction")
        if not 1 <= count <= size:
            raise ValueError(f"epoch_size count must be in [1, {size}]")

        initialized = distributed.is_available() and distributed.is_initialized()
        if num_replicas is None:
            num_replicas = distributed.get_world_size() if initialized else 1
        if rank is None:
            rank = distributed.get_rank() if initialized else 0
        num_replicas = validate_integer(num_replicas, "num_replicas", 1)
        rank = validate_integer(rank, "rank", 0)
        if rank >= num_replicas:
            raise ValueError("rank must be smaller than num_replicas")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool")
        batch_size = validate_integer(batch_size, "batch_size", 1)
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool")
        if balance is not None and not isinstance(balance, str):
            try:
                balance = tuple(
                    validate_integer(w, "balance weight", 0) for w in balance
                )
            except (TypeError, ValueError):
                raise ValueError(
                    "balance weights must be two nonnegative integers"
                ) from None
            if len(balance) != 2 or not any(balance):
                raise ValueError(
                    "balance weights must be two integers with a positive sum"
                )
            if max(balance) > np.iinfo(np.int64).max:
                raise ValueError("balance weights exceed the int64 range")
        elif balance not in (None, "atoms", "edges"):
            raise ValueError("balance must be None, 'atoms', 'edges' or two weights")
        if balance is not None and not callable(getattr(dataset, "get_sizes", None)):
            raise ValueError("Balanced sampling requires cached graph sizes")
        seed = validate_integer(seed, "seed")
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.dataset_size = size
        self.dataset_fingerprint = getattr(dataset, "sample_fingerprint", None)
        self.epoch_size = count
        self.batch_size = batch_size
        self.balance = balance
        self._costs = None
        multiple = num_replicas * (batch_size if drop_last else 1)
        self.total_size = ((count + multiple - 1) // multiple) * multiple
        self.num_samples = self.total_size // num_replicas
        self.consumed = 0
        self.start = 0
        self._half_bits = ((size - 1).bit_length() + 1) // 2
        self._half_mask = np.uint64((1 << self._half_bits) - 1)
        self._round_keys = tuple(
            np.uint64((seed + (i + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
            for i in range(6)
        )

    def _feistel(self, values):
        left = values >> self._half_bits
        right = values & self._half_mask
        for key in self._round_keys:
            mixed = right + key
            mixed = (mixed ^ (mixed >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
            mixed = (mixed ^ (mixed >> 27)) * np.uint64(0x94D049BB133111EB)
            mixed ^= mixed >> 31
            left, right = right, left ^ (mixed & self._half_mask)
        return (left << self._half_bits) | right

    def _indices(self, start_position, consumed):
        if self.balance is not None and self.num_replicas > 1:
            yield from self._balanced_indices(start_position, consumed)
            return
        base = start_position % self.dataset_size
        for start in range(consumed, self.num_samples, self._chunk_size):
            stop = min(start + self._chunk_size, self.num_samples)
            positions = np.arange(start, stop, dtype=np.uint64)
            positions = (
                positions * self.num_replicas + self.rank + base
            ) % self.dataset_size
            if self.shuffle:
                positions = self._feistel(positions)
                outside = positions >= self.dataset_size
                while outside.any():
                    positions[outside] = self._feistel(positions[outside])
                    outside = positions >= self.dataset_size
            yield from map(int, positions)

    def _balanced_indices(self, start_position, consumed):
        if self._costs is None:
            sizes = self.dataset.get_sizes()
            if isinstance(self.balance, tuple):
                maximum = sum(
                    int(n) * w for n, w in zip(sizes.max(axis=0), self.balance)
                )
                if maximum > np.iinfo(np.int64).max:
                    raise ValueError("Weighted graph sizes exceed the int64 range")
                self._costs = sizes @ np.asarray(self.balance, dtype=np.int64)
            else:
                column = 0 if self.balance == "atoms" else 1
                self._costs = sizes[:, column].copy()
            del sizes
        base = start_position % self.dataset_size
        first = consumed // self.batch_size * self.batch_size
        for start in range(first, self.num_samples, self.batch_size):
            stop = min(start + self.batch_size, self.num_samples)
            positions = np.arange(
                start * self.num_replicas, stop * self.num_replicas, dtype=np.uint64
            )
            positions = (positions + base) % self.dataset_size
            if self.shuffle:
                positions = self._feistel(positions)
                outside = positions >= self.dataset_size
                while outside.any():
                    positions[outside] = self._feistel(positions[outside])
                    outside = positions >= self.dataset_size
            costs = self._costs[positions]
            bins = [[] for _ in range(self.num_replicas)]
            rotation = (self.seed + start // self.batch_size) % self.num_replicas
            heap = [
                (0, 0, r, (r + rotation) % self.num_replicas)
                for r in range(self.num_replicas)
            ]
            loads = [0] * self.num_replicas
            for index in np.argsort(-costs, kind="stable"):
                load, count, order, rank = heapq.heappop(heap)
                bins[rank].append(int(index))
                loads[rank] = load + int(costs[index])
                if count + 1 < stop - start:
                    heapq.heappush(heap, (loads[rank], count + 1, order, rank))
            # Cardinality constraints can make greedy packing worse. Retain
            # the original partition in that case.
            original = costs.reshape(-1, self.num_replicas).sum(axis=0)
            if max(loads) > original.max():
                selected = positions[self.rank :: self.num_replicas]
            else:
                selected = positions[sorted(bins[self.rank])]
            yield from map(int, selected[max(0, consumed - start) :])

    def __iter__(self):
        # Capture at iter(), not first next(), so later commits cannot skip work.
        return self._indices(self.start, self.consumed)

    def set_epoch(self, epoch):
        """Select an epoch without skipping an interrupted predecessor's tail.

        Consecutive epochs continue after committed samples. An unconsumed
        sampler, or an explicit jump across epochs, selects the corresponding
        deterministic window without requiring preceding epochs to be read.
        """
        epoch = validate_integer(epoch, "epoch", 0)
        if epoch != self.epoch:
            if epoch == self.epoch + 1 and self.consumed:
                if (
                    self.balance is not None
                    and self.num_replicas > 1
                    and self.consumed % self.batch_size
                    and self.consumed != self.num_samples
                ):
                    raise ValueError(
                        "Finish the partially consumed balanced batch before "
                        "advancing to the next epoch"
                    )
                self.start += self.consumed * self.num_replicas
            else:
                self.start = epoch * self.total_size
            self.epoch = epoch
            self.consumed = 0

    def advance(self, count):
        """Commit ``count`` local samples after their training batch completes."""
        count = validate_integer(count, "count", 0)
        if self.consumed + count > self.num_samples:
            raise ValueError("Committed sample count exceeds this local epoch")
        self.consumed += count

    def state_dict(self):
        """Return rank-independent state suitable for a rank-zero checkpoint."""
        return {
            "dataset_size": self.dataset_size,
            "dataset_fingerprint": self.dataset_fingerprint,
            "epoch_size": self.epoch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "num_replicas": self.num_replicas,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "balance": self.balance,
            "epoch": self.epoch,
            "consumed": self.consumed,
            "start": self.start,
        }

    def load_state_dict(self, state):
        """Restore progress, rejecting changed dataset or sampling parameters."""
        current = self.state_dict()
        if state.get("balance") != self.balance:
            raise ValueError("Cannot restore EpochSampler with changed balance")
        if state.get("dataset_fingerprint") != self.dataset_fingerprint:
            raise ValueError(
                "Cannot restore EpochSampler with changed dataset selection"
            )
        for name in (
            "dataset_size",
            "epoch_size",
            "seed",
            "num_replicas",
            "batch_size",
        ):
            value = validate_integer(state[name], name)
            if value != current[name]:
                raise ValueError(f"Cannot restore EpochSampler with changed {name}")
        for name in ("shuffle", "drop_last"):
            if not isinstance(state[name], bool):
                raise TypeError(f"{name} must be a bool")
            if state[name] != current[name]:
                raise ValueError(f"Cannot restore EpochSampler with changed {name}")
        epoch = validate_integer(state["epoch"], "epoch", 0)
        consumed = validate_integer(state["consumed"], "consumed", 0)
        start = validate_integer(state["start"], "start", 0)
        if consumed > self.num_samples:
            raise ValueError("Committed sample count exceeds this local epoch")
        self.epoch = epoch
        self.consumed = consumed
        self.start = start
