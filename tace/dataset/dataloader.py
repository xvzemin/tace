################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from .sampler import EpochSampler
from .store import GraphDatasetLMDB, decode_array_record


class StatisticsDataset(GraphDatasetLMDB):
    """Decode owned field tensors without constructing one PyG object per graph."""

    def _decode(self, value):
        return decode_array_record(value, self.kind, self.fields)


class StatisticsDataLoader(DataLoader):
    """Load projected graph fields for statistics reductions.

    Parameters
    ----------
    dataset : Dataset
        Graph dataset. Disk LMDB fields are decoded directly into tensors.
    **kwargs
        Arguments to :class:`torch_geometric.loader.DataLoader`.

    Notes
    -----
    Batches include ``ptr`` and ``batch`` but omit reconstruction metadata for
    ``Batch.to_data_list()``.

    Examples
    --------
    >>> loader = StatisticsDataLoader(dataset, batch_size=16384, num_workers=0)
    """

    def __init__(self, dataset, **kwargs):
        if (
            isinstance(dataset, GraphDatasetLMDB)
            and not dataset.in_memory
            and not kwargs.get("follow_batch")
            and not kwargs.get("exclude_keys")
        ):
            source = dataset
            dataset = StatisticsDataset(
                source.lmdb_paths,
                shard_lengths=source.lengths,
                fields=source.fields,
                cache_size=0,
                max_open_shards=source.max_open_shards,
                readahead=source.readahead,
            )
            dataset.metadata = source.metadata
            dataset.indices = source.indices
            dataset.sample_fingerprint = source.sample_fingerprint
        super().__init__(dataset, **kwargs)
        if not self.follow_batch and not self.exclude_keys:
            self.collate_fn = collate_statistics


def collate_statistics(graphs):
    """Concatenate homogeneous, projected graph fields for FP64 statistics."""
    counts = np.fromiter((graph["num_nodes"] for graph in graphs), dtype=np.int64)
    ptr = np.concatenate((np.zeros(1, dtype=np.int64), counts.cumsum()))
    batch = Batch(
        num_nodes=int(ptr[-1]),
        ptr=torch.from_numpy(ptr),
        batch=torch.from_numpy(
            np.repeat(np.arange(len(graphs), dtype=np.int64), counts)
        ),
    )
    for key in graphs[0].keys():
        if key == "num_nodes":
            continue
        values = [graph[key] for graph in graphs]
        if key == "edge_index":
            batch[key] = torch.cat(
                [value + int(offset) for value, offset in zip(values, ptr[:-1])],
                dim=1,
            )
        else:
            batch[key] = torch.cat(
                [value.reshape(1) if value.ndim == 0 else value for value in values],
                dim=0,
            )
    return batch


class EpochDataLoader(DataLoader):
    """Graph loader with checkpointable, consumed-sample progress.

    Parameters
    ----------
    dataset : Dataset
        Graph dataset.
    sampler : EpochSampler
        Training sampler.
    **kwargs
        Arguments to :class:`torch_geometric.loader.DataLoader`.
    """

    def __iter__(self):
        # Lightning sets the epoch after constructing the loader iterator.
        yield from super().__iter__()

    def mark_batch_complete(self):
        self.sampler.advance(
            min(self.batch_size, self.sampler.num_samples - self.sampler.consumed)
        )

    def state_dict(self):
        return {
            "sampler": self.sampler.state_dict(),
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
        }

    def load_state_dict(self, state):
        for name in ("batch_size", "drop_last"):
            if state[name] != getattr(self, name):
                raise ValueError(
                    f"Cannot resume epoch sampling with a different {name}."
                )
        self.sampler.load_state_dict(state["sampler"])


def build_graph_dataloader(
    config, dataset, *, epoch_size=None, seed=0, balance_batches=None
):
    """Build a loader without bypassing batched dataset reads."""
    config = {key: value for key, value in config.items() if key != "extra"}
    if balance_batches is not None and epoch_size is None:
        raise ValueError("balance_batches requires epoch_size (use 1.0 for all graphs)")
    source = getattr(dataset, "dataset", dataset)
    device = getattr(source, "device", None)
    if (
        device is not None
        and torch.device(device).type == "cuda"
        and config.get("num_workers", 0)
    ):
        raise ValueError("CUDA online graph construction requires num_workers=0.")
    if epoch_size is not None:
        if config.get("_target_") not in {
            "torch_geometric.loader.DataLoader",
            "torch_geometric.data.DataLoader",
            "tace.dataset.dataloader.EpochDataLoader",
        }:
            raise ValueError("epoch_size requires a torch_geometric DataLoader.")
        if config.get("sampler") is not None or config.get("batch_sampler") is not None:
            raise ValueError("epoch_size cannot be combined with another sampler.")
        if config.get("in_order", True) is not True:
            raise ValueError(
                "epoch_size requires in_order=True for checkpoint recovery."
            )
        sampler = EpochSampler(
            dataset,
            epoch_size,
            shuffle=config.get("shuffle", False),
            seed=seed,
            batch_size=config.get("batch_size", 1),
            drop_last=config.get("drop_last", False),
            balance=balance_batches,
        )
        if sampler.rank == 0 and sampler.total_size != sampler.epoch_size:
            logging.info(
                "Training epoch size: requested %d, actual %d graphs "
                "(%d ranks, batch_size=%d, drop_last=%s); continuing the data stream.",
                sampler.epoch_size,
                sampler.total_size,
                sampler.num_replicas,
                sampler.batch_size,
                sampler.drop_last,
            )
        config.update(
            _target_="tace.dataset.dataloader.EpochDataLoader",
            sampler=sampler,
            shuffle=False,
        )
    return instantiate(config, dataset=dataset)


def build_statistics_dataloader(
    cfg: Dict,
    dataset,
    target_property: Sequence[str] = (),
    neighbors: bool = True,
):
    """Build a large, deterministic loader containing only statistics inputs."""
    config = {
        key: value
        for key, value in cfg["dataset"]["train_dataloader"].items()
        if key != "extra"
    }
    statistics_config = cfg["dataset"].get("statistics_dataloader", {})
    config.update(statistics_config)
    config.pop("extra", None)
    # Large statistics batches should not inherit training-time prefetch queues.
    config["num_workers"] = statistics_config.get("num_workers", 0)
    if config["num_workers"]:
        config["prefetch_factor"] = statistics_config.get("prefetch_factor", 1)
    else:
        config.pop("prefetch_factor", None)
        config.pop("persistent_workers", None)
        config.pop("multiprocessing_context", None)
    config["shuffle"] = False
    config["drop_last"] = False
    config.setdefault(
        "pin_memory",
        str(cfg.get("misc", {}).get("device", "cpu")).startswith("cuda"),
    )

    required_keys = {
        "atomic_numbers",
        "fidelity_idx",
        "initial_noncollinear_magmoms",
        "num_nodes",
    }
    if neighbors:
        fields = (getattr(dataset, "metadata", None) or {}).get("fields", [])
        required_keys.add(
            "num_neighbors" if "num_neighbors" in fields else "edge_index"
        )
    if "energy" in target_property:
        required_keys.update(("energy", "energy_weight"))
    if "forces" in target_property:
        required_keys.update(("forces", "forces_weight"))
    if "direct_forces" in target_property:
        required_keys.update(("direct_forces", "direct_forces_weight"))
    if hasattr(dataset, "project"):
        dataset = dataset.project(required_keys)
        config.pop("exclude_keys", None)
    elif len(dataset):
        excluded_keys = set(config.get("exclude_keys") or ())
        excluded_keys.update(set(dataset[0].keys()) - required_keys)
        config["exclude_keys"] = sorted(excluded_keys - required_keys)
    if isinstance(dataset, GraphDatasetLMDB) and config.get("_target_") in {
        "torch_geometric.loader.DataLoader",
        "torch_geometric.data.DataLoader",
        "tace.dataset.dataloader.StatisticsDataLoader",
    }:
        config["_target_"] = "tace.dataset.dataloader.StatisticsDataLoader"
    return build_graph_dataloader(config, dataset)
