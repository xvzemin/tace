################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Read-only selection of cached graphs before training and epoch sampling."""

import logging
import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from .element import ATOMIC_COVALENT_RADII
from .store import GraphDatasetLMDB

_DEFAULTS = {
    "max_edges": None,
    "max_atoms": None,
    "max_forces": None,
    "max_energy_per_atom": None,
    "max_stress": None,
    "min_covalent_ratio": None,
    "drop_isolated": False,
}
_EDGE_GEOMETRY = ("positions", "edge_index", "edge_shifts", "lattice")
_RADII = torch.tensor(ATOMIC_COVALENT_RADII, dtype=torch.float64, device="cpu")
_LABELS = {
    "max_forces": "forces",
    "max_energy_per_atom": "energy",
    "max_stress": "stress",
}


def filter_options(config):
    """Validate ``dataset.filter`` and return its enabled filters.

    ``drop_isolated`` drops multi-atom structures if any atom has no neighbor
    within the graph cutoff. Single-atom structures are kept by this filter.
    """
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
    options = {}
    for name, default in _DEFAULTS.items():
        value = config.get(name, default)
        if name == "drop_isolated":
            if not isinstance(value, bool):
                raise TypeError("dataset.filter.drop_isolated must be a bool")
            if value:
                options[name] = True
            continue
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
    return options


def size_chunks(dataset):
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


def edge_contacts(graph):
    """Return directed edges, periodic distances and zero-shift self loops."""
    positions, indices, shifts, lattice = (graph[key] for key in _EDGE_GEOMETRY)
    source, target = indices
    positions = positions.to(torch.float64)
    vectors = positions.index_select(0, target) - positions.index_select(0, source)
    vectors.addmm_(shifts.to(positions), lattice.reshape(3, 3).to(positions))
    distances = torch.linalg.vector_norm(vectors, dim=-1)
    self_loops = (source == target) & (shifts == 0).all(dim=-1)
    return source, target, distances, self_loops


def check_covalent_cutoff(dataset, ratio: float, cutoff):
    """Check that the cached neighbor list covers the requested element pairs."""
    if cutoff is None:
        raise ValueError(
            "dataset.filter.min_covalent_ratio requires a known graph cutoff"
        )
    if 2 * ratio * max(ATOMIC_COVALENT_RADII) <= cutoff:
        return
    elements = (dataset.metadata or {}).get("atomic_numbers")
    if elements is None:
        scan = dataset.project(None).project(["atomic_numbers"])
        try:
            elements = {z for graph in scan for z in graph.atomic_numbers.tolist()}
        finally:
            scan.close()
    required = (
        2 * ratio * max((ATOMIC_COVALENT_RADII[z] for z in elements), default=0.0)
    )
    if required > cutoff:
        raise ValueError(
            f"dataset.filter.min_covalent_ratio={ratio} requires a graph cutoff "
            f"of at least {required:g} angstroms for these elements (cutoff={cutoff:g})"
        )


def accepts_graph(graph, bounds):
    if bounds.get("drop_isolated") and graph.num_nodes > 1:
        degrees = graph.get("num_neighbors")
        if degrees is None:
            source, target = graph.edge_index
            neighbor = (source != target) | graph.edge_shifts.any(dim=-1)
            degrees = torch.bincount(source[neighbor], minlength=graph.num_nodes)
        if not bool((degrees > 0).all()):
            return False
    minimum = bounds.get("min_covalent_ratio")
    if minimum is not None:
        source, target, distances, self_loops = edge_contacts(graph)
        radii = _RADII.index_select(0, graph.atomic_numbers)
        lower = (radii[source] + radii[target]) * minimum
        if not bool(
            (self_loops | (torch.isfinite(distances) & (distances >= lower))).all()
        ):
            return False
    for name, limit in bounds.items():
        if name not in _LABELS:
            continue
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


def filter_graphs(dataset, options, stage="train", *, graph_cutoff=None):
    """Select graphs satisfying every bound without changing cached records.

    Atom and edge counts come from the shard count index. Other filters read
    only the required degrees, geometry or labels and weights. Periodic
    distances use cached lattice shifts. ``graph_cutoff`` validates
    ``min_covalent_ratio`` when the cutoff is absent from the shard metadata.
    ``options`` is the result of :func:`filter_options`.
    """
    if not options:
        return dataset
    if not isinstance(dataset, GraphDatasetLMDB):
        raise ValueError(
            "dataset.filter requires storage_mode='lmdb', 'clmdb' or 'memory'"
        )
    bounds = {k: v for k, v in options.items() if k not in {"max_atoms", "max_edges"}}
    fields = {_LABELS[name] for name in bounds if name in _LABELS}
    has_labels = bool(fields)
    fields.update(f"{name}_weight" for name in tuple(fields))
    metadata = dataset.metadata or {}
    available = metadata.get("fields")
    drop_isolated = bounds.get("drop_isolated", False)
    if drop_isolated:
        if available is None and dataset.in_memory and dataset.data_list:
            available = dataset.data_list[0].keys()
        if available is not None and "num_neighbors" in available:
            fields.add("num_neighbors")
        else:
            fields.update(("edge_index", "edge_shifts"))
    if "min_covalent_ratio" in bounds:
        cutoff = (metadata.get("graph_config") or {}).get("cutoff", graph_cutoff)
        check_covalent_cutoff(dataset, bounds["min_covalent_ratio"], cutoff)
        fields.update((*_EDGE_GEOMETRY, "atomic_numbers"))
    if available is not None and fields - set(available):
        raise ValueError(
            f"Graph shards lack fields required by dataset.filter: {sorted(fields - set(available))}"
        )
    keep = np.ones(len(dataset), dtype=bool)
    scan = dataset.project(None).project(fields) if fields else None
    disabled = dist.is_available() and dist.is_initialized() and dist.get_rank() != 0
    try:
        with tqdm(
            total=len(dataset), desc=f"{stage} filter", unit="graph", disable=disabled
        ) as progress:
            for start, sizes in size_chunks(dataset):
                mask = keep[start : start + len(sizes)]
                if "max_atoms" in options:
                    mask &= sizes[:, 0] <= options["max_atoms"]
                if "max_edges" in options:
                    mask &= sizes[:, 1] <= options["max_edges"]
                atoms, edges = sizes.T
                if drop_isolated:
                    # Directed lists need at least one outgoing edge per atom.
                    mask &= (atoms == 1) | ((atoms > 1) & (edges >= atoms))
                if scan is not None:
                    to_scan = mask
                    if not has_labels:
                        to_scan = mask & (edges != 0)
                        if drop_isolated and "min_covalent_ratio" not in bounds:
                            to_scan &= atoms > 1
                    candidates = np.flatnonzero(to_scan) + start
                    for offset in range(0, len(candidates), 256):
                        indices = candidates[offset : offset + 256]
                        graphs = scan.__getitems__(indices)
                        keep[indices] = [
                            accepts_graph(graph, bounds) for graph in graphs
                        ]
                progress.update(len(sizes))
    finally:
        if scan is not None:
            scan.close()
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
