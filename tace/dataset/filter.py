"""Read-only selection of cached graphs before training and epoch sampling."""

import logging
import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from .store import GraphDatasetLMDB

_DEFAULTS = {
    "max_edges": 24000,
    "max_atoms": None,
    "max_forces": None,
    "max_energy_per_atom": None,
    "max_stress": None,
}
_LABELS = {
    "max_forces": "forces",
    "max_energy_per_atom": "energy",
    "max_stress": "stress",
}


def filter_options(config):
    """Validate ``dataset.filter`` and return its enabled upper bounds."""
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("dataset.filter must be a mapping")
    unknown = set(config) - {"enabled", *_DEFAULTS}
    if unknown:
        raise ValueError(f"Unknown dataset.filter options: {sorted(unknown)}")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("dataset.filter.enabled must be a bool")
    if not enabled:
        return {}
    options = {**_DEFAULTS, **{k: v for k, v in config.items() if k != "enabled"}}
    for name, value in options.items():
        if value is None:
            continue
        count = name in {"max_edges", "max_atoms"}
        if isinstance(value, bool) or not isinstance(
            value, Integral if count else Real
        ):
            raise TypeError(
                f"dataset.filter.{name} must be a finite {'integer' if count else 'number'} or null"
            )
        if not math.isfinite(value) or (name != "max_energy_per_atom" and value < 0):
            raise ValueError(f"Invalid dataset.filter.{name}: {value}")
        options[name] = int(value) if count else float(value)
    return {name: value for name, value in options.items() if value is not None}


def _size_chunks(dataset):
    # Bound temporary memory even for tens of millions of cached structures.
    chunk_size = 65536
    if dataset.in_memory or dataset.indices is not None:
        for start in range(0, len(dataset), chunk_size):
            yield (
                start,
                dataset.get_sizes(range(start, min(start + chunk_size, len(dataset)))),
            )
        return
    offset = 0
    for path, length in zip(dataset.lmdb_paths, dataset.lengths):
        if length:
            counts = np.memmap(
                path / "counts.bin", mode="r", dtype="<i8", shape=(length, 2)
            )
            for start in range(0, length, chunk_size):
                yield offset + start, counts[start : start + chunk_size]
            del counts
        offset += length


def _accepts(graph, bounds):
    for name, limit in bounds.items():
        field = _LABELS[name]
        value = graph.get(field)
        weight = graph.get(f"{field}_weight")
        if value is None or weight is None:
            raise ValueError(
                f"Graph lacks {field} or {field}_weight required by dataset.filter.{name}"
            )
        # Missing labels are stored as placeholders with zero weight; they must
        # not pass a physical threshold as if they were measured zeros.
        if not bool(torch.isfinite(weight).all() and (weight > 0).all()):
            return False
        if not value.numel() or not bool(torch.isfinite(value).all()):
            return False
        if name == "max_forces":
            metric = torch.linalg.vector_norm(value, dim=-1).max().item()
        elif name == "max_energy_per_atom":
            if not graph.num_nodes:
                return False
            metric = value.item() / graph.num_nodes
        else:
            metric = torch.linalg.vector_norm(value.reshape(-1)).item()
        if metric > limit:
            return False
    return True


def filter_graphs(dataset, options, stage="train"):
    """Select graphs satisfying every bound without changing cached records.

    Atom and edge counts come from the shard count index. Physical filters read
    only their labels and weights, once during setup. Limits are inclusive;
    energy uses signed total energy per atom, forces use the maximum atom-wise
    Euclidean norm, and stress uses the full tensor's Frobenius norm.
    ``options`` is the result of :func:`filter_options`.
    """
    if not options:
        return dataset
    if not isinstance(dataset, GraphDatasetLMDB):
        raise ValueError(
            "dataset.filter requires storage_mode='lmdb', 'clmdb' or 'memory'"
        )
    bounds = {name: value for name, value in options.items() if name in _LABELS}
    fields = {_LABELS[name] for name in bounds}
    fields.update(f"{name}_weight" for name in tuple(fields))
    available = (dataset.metadata or {}).get("fields")
    if available is not None and fields - set(available):
        raise ValueError(
            f"Graph shards lack fields required by dataset.filter: {sorted(fields - set(available))}"
        )
    keep = np.ones(len(dataset), dtype=bool)
    # A filter can need a cached label that is not a current training target.
    # Restore it only for the scan; selected samples retain their projection.
    labels = dataset.project(None).project(fields) if bounds else None
    disabled = dist.is_available() and dist.is_initialized() and dist.get_rank() != 0
    try:
        with tqdm(
            total=len(dataset), desc=f"{stage} filter", unit="graph", disable=disabled
        ) as progress:
            for start, sizes in _size_chunks(dataset):
                mask = keep[start : start + len(sizes)]
                if "max_atoms" in options:
                    mask &= sizes[:, 0] <= options["max_atoms"]
                if "max_edges" in options:
                    mask &= sizes[:, 1] <= options["max_edges"]
                if bounds:
                    candidates = np.flatnonzero(mask) + start
                    for offset in range(0, len(candidates), 256):
                        indices = candidates[offset : offset + 256]
                        graphs = labels.__getitems__(indices)
                        keep[indices] = [_accepts(graph, bounds) for graph in graphs]
                progress.update(len(sizes))
    finally:
        if labels is not None:
            labels.close()
    count = int(np.count_nonzero(keep))
    if not count:
        raise ValueError(
            f"dataset.filter removed all {len(dataset)} {stage} structures"
        )
    selected = (
        dataset if count == len(dataset) else dataset.select(np.flatnonzero(keep))
    )
    logging.info(
        "%s filter: kept %d / %d structures; removed %d (%s)",
        stage,
        len(selected),
        len(dataset),
        len(dataset) - len(selected),
        options,
    )
    return selected
