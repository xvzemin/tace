################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Read-only selection of cached graphs before training and epoch sampling."""

import logging
import math
import multiprocessing
import os
from bisect import bisect_right
from collections.abc import Mapping
from itertools import chain, tee
from numbers import Integral, Real

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .element import ATOMIC_COVALENT_RADII
from .store import GraphDatasetLMDB, decode_array_record
from .topology import passes_topology

_DEFAULTS = {
    "max_edges": None,
    "max_atoms": None,
    "max_forces": None,
    "max_energy_per_atom": None,
    "max_stress": None,
    "min_covalent_ratio": None,
    "drop_isolated": False,
    "drop_disconnected": True,
    "max_rotation_forces_rmse": 1e-4,
    "max_translation_forces_rmse": 0.01,
    "max_forces_stress_residual": 1e-6,
}
_RIGID_FORCES = {"max_rotation_forces_rmse", "max_translation_forces_rmse"}
_EDGE_GEOMETRY = ("positions", "edge_index", "edge_shifts", "lattice")
_RADII = torch.tensor(ATOMIC_COVALENT_RADII, dtype=torch.float64, device="cpu")
_LABELS = {
    "max_forces": "forces",
    "max_energy_per_atom": "energy",
    "max_stress": "stress",
}
FILTER_ROWS_PER_RANK = 16384


def filter_options(config):
    """Validate ``dataset.filter`` and return its enabled filters.

    Enabled filters check connectivity, rigid-motion forces and the constrained
    stress residual by default.
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
        if name in {"drop_isolated", "drop_disconnected"}:
            if not isinstance(value, bool):
                raise TypeError(f"dataset.filter.{name} must be a bool")
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


def filter_chunks(dataset, first, stop, chunk_size):
    """Yield view positions and shard-local slices or selected record indices."""
    if first == stop:
        return
    selected = dataset.indices
    ends = [dataset.length] if dataset.in_memory else dataset._ends
    begin = bisect_right(ends, first if selected is None else selected[first])
    for shard in range(begin, len(ends)):
        offset = ends[shard - 1] if shard else 0
        lo, hi = offset, ends[shard]
        if selected is not None:
            lo, hi = selected.searchsorted((lo, hi))
        if lo >= stop:
            break
        for start in range(max(first, lo), min(stop, hi), chunk_size):
            end = min(stop, hi, start + chunk_size)
            if selected is None:
                items = slice(start - offset, end - offset)
            elif selected[end - 1] - selected[start] == end - start - 1:
                items = slice(selected[start] - offset, selected[end - 1] - offset + 1)
            else:
                items = selected[start:end] - offset
            yield start, shard, items


def passes_covalent_ratio(graph, minimum):
    """Check periodic contacts, excluding only zero-shift self loops."""
    positions, indices, shifts, lattice = (graph[key] for key in _EDGE_GEOMETRY)
    source, target = indices
    if not source.numel():
        return True
    positions = positions.to(torch.float64)
    vectors = positions.index_select(0, target) - positions.index_select(0, source)
    vectors.addmm_(shifts.to(positions), lattice.reshape(3, 3).to(positions))
    distances = torch.linalg.vector_norm(vectors, dim=-1)
    radii = _RADII.index_select(0, graph["atomic_numbers"])
    lower = (radii.index_select(0, source) + radii.index_select(0, target)) * minimum
    invalid = ~(torch.isfinite(distances) & (distances >= lower))
    if not invalid.any():
        return True
    return bool(
        (
            (source[invalid] == target[invalid]) & (shifts[invalid] == 0).all(dim=-1)
        ).all()
    )


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


def isolated_reference(graph, cutoff):
    """Whether a one-site graph has no active periodic self image."""
    shifts = graph.get("edge_shifts")
    if shifts is None:
        edges = graph.get("edge_index")
        return edges is not None and edges.shape[1] == 0
    shifts = np.asarray(shifts)
    if not shifts.any():
        return True
    if cutoff is None or graph.get("lattice") is None:
        return False
    vectors = shifts[shifts.any(axis=1)] @ np.asarray(graph["lattice"]).reshape(3, 3)
    return bool(np.all(np.einsum("ij,ij->i", vectors, vectors) >= cutoff**2))


def accepts_graph(graph, bounds, num_nodes, cutoff=None, missing=()):
    if num_nodes == 1 and isolated_reference(graph, cutoff):
        return True
    if missing:
        raise ValueError(
            f"Graph shards lack fields required by dataset.filter: {sorted(missing)}"
        )
    if bounds.get("drop_isolated") and num_nodes > 1:
        degrees = graph.get("num_neighbors")
        if degrees is None:
            source, target = np.asarray(graph["edge_index"])
            neighbor = (source != target) | np.asarray(graph["edge_shifts"]).any(
                axis=-1
            )
            degrees = np.bincount(source[neighbor], minlength=num_nodes)
        if not (np.asarray(degrees) > 0).all():
            return False
    minimum = bounds.get("min_covalent_ratio")
    if minimum is not None and not passes_covalent_ratio(graph, minimum):
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
        if weight.numel() == 1:
            w = weight.item()
            if not (math.isfinite(w) and w > 0):
                return False
        elif not bool((torch.isfinite(weight) & (weight > 0)).all()):
            return False
        if not value.numel():
            return False
        if name == "max_forces":
            metric = torch.linalg.vector_norm(value, dim=-1).max().item()
        elif name == "max_energy_per_atom":
            if not num_nodes:
                return False
            metric = value.item() / num_nodes
        else:
            metric = torch.linalg.vector_norm(value.reshape(-1)).item()
        if not math.isfinite(metric) or metric > limit:
            return False
    if (
        "drop_disconnected" in bounds
        or not _RIGID_FORCES.isdisjoint(bounds)
        or "max_forces_stress_residual" in bounds
    ):
        return passes_topology(
            graph,
            num_nodes,
            cutoff,
            bounds.get("drop_disconnected", False),
            bounds.get("max_rotation_forces_rmse"),
            bounds.get("max_translation_forces_rmse"),
            bounds.get("max_forces_stress_residual"),
        )
    return True


class FilterScan(GraphDatasetLMDB):
    """Evaluate filter predicates on a shard; return only the selection mask."""

    def __init__(self, dataset, options, bounds, fields, cutoff, missing):
        super().__init__(
            dataset.lmdb_paths,
            in_memory=dataset.in_memory,
            shard_lengths=dataset.lengths,
            fields=fields,
            readahead=True,
            max_open_shards=2,
        )
        self.data_list = dataset.data_list
        self.options = options
        self.bounds = bounds
        self.cutoff = cutoff
        self.missing = missing
        self.numpy_views = not (
            _LABELS.keys() & bounds.keys() or "min_covalent_ratio" in bounds
        )
        self.has_labels = bool(_LABELS.keys() & bounds.keys()) or (
            "max_translation_forces_rmse" in bounds
            or "max_forces_stress_residual" in bounds
        )

    def _decode(self, value):
        record = decode_array_record(
            value, self.kind, self.fields, numpy_views=self.numpy_views
        )
        return accepts_graph(
            record, self.bounds, record["num_nodes"], self.cutoff, self.missing
        )

    def decode_singleton(self, value):
        graph = decode_array_record(
            value, self.kind, ("edge_shifts", "lattice"), numpy_views=True
        )
        return isolated_reference(graph, self.cutoff)

    def counts(self, shard):
        return np.memmap(
            self.lmdb_paths[shard] / "counts.bin",
            mode="r",
            dtype="<i8",
            shape=(self.lengths[shard], 2),
        )

    def __getitem__(self, job):
        start, shard, items = job
        if self.in_memory:
            indices = (
                range(items.start, items.stop) if isinstance(items, slice) else items
            )
            sizes = np.asarray(
                [
                    (self.data_list[i].num_nodes, self.data_list[i].num_edges)
                    for i in indices
                ]
            )
        else:
            sizes = self.counts(shard)[items]
        atoms, edges = sizes.T
        keep = np.ones(len(atoms), dtype=bool)
        if "max_atoms" in self.options:
            keep &= atoms <= self.options["max_atoms"]
        if "max_edges" in self.options:
            keep &= edges <= self.options["max_edges"]
        drop = self.bounds.get("drop_isolated", False)
        if drop:
            keep &= (atoms == 1) | ((atoms > 1) & (edges >= atoms))
        if self.bounds.get("drop_disconnected"):
            keep &= edges >= atoms - 1
        protected = (atoms == 1) & (edges == 0)
        # Singleton count failures may still be isolated if all stored edges
        # are zero-shift loops or exactly at the cutoff.
        ambiguous = (atoms == 1) & ~protected & ~keep
        if self.bounds:
            scan = (keep & ~protected) if self.has_labels else (keep & (edges != 0))
            rows = np.flatnonzero(scan)
            indices = items.start + rows if isinstance(items, slice) else items[rows]
            if self.in_memory:
                keep[rows] = [
                    accepts_graph(
                        self.data_list[i], self.bounds, n, self.cutoff, self.missing
                    )
                    for i, n in zip(indices, atoms[rows])
                ]
            elif len(rows):
                keep[rows] = self.map_shard(shard, indices, self._decode)
        if ambiguous.any():
            rows = np.flatnonzero(ambiguous)
            indices = items.start + rows if isinstance(items, slice) else items[rows]
            keep[rows] = (
                [isolated_reference(self.data_list[i], self.cutoff) for i in indices]
                if self.in_memory
                else self.map_shard(shard, indices, self.decode_singleton)
            )
        keep |= protected
        return start, keep


def filter_workers(size, world, context):
    """Choose CPU readers per rank for a disk scan."""
    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    if multiprocessing.current_process().daemon:
        return 0
    local_world = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", world)))
    minimum = 8192 if context == "fork" else 131072
    # Budget readers per host, not independently for each training rank.
    budget = 48 if context in ("fork", "forkserver") else 8
    limit = max(1, budget // local_world)
    if context != "fork":
        limit = min(limit, 8)  # Each fresh interpreter imports its own torch runtime.
    cores //= local_world
    return min(limit, cores, size // 2048) if size >= minimum and cores > 1 else 0


def scan_jobs(scan, jobs, workers, context):
    if workers:
        # An isolated-reference prefix (or an entirely atomic dataset) needs
        # neither a graph decode nor a torch reader pool.
        last_shard = None
        for job in jobs:
            start, shard, items = job
            if shard != last_shard:
                counts, last_shard = scan.counts(shard), shard
            atoms, edges = counts[items].T
            if (atoms == 1).all() and not edges.any():
                yield start, np.ones(len(atoms), dtype=bool)
            else:
                jobs = chain((job,), jobs)
                break
        else:
            return
        # Keep torch's training RNG and CUDA context out of the reader pool.
        yield from DataLoader(
            scan,
            batch_size=None,
            sampler=jobs,
            num_workers=workers,
            collate_fn=tuple,
            generator=torch.Generator().manual_seed(0),
            prefetch_factor=1,
            multiprocessing_context=context,
        )
    else:
        yield from map(scan.__getitem__, jobs)


def scan_ranges(dataset, scan, ranges, world, advance=None):
    """Scan logical ranges with one CPU reader pool for the entire traversal."""
    size = 1024 if scan.bounds else 65536
    ranges, work = tee(ranges)
    jobs = (job for lo, hi in work for job in filter_chunks(dataset, lo, hi, size))
    workers, context = 0, None
    if scan.bounds and not dataset.in_memory:
        context = (
            multiprocessing.get_start_method(allow_none=True)
            or multiprocessing.get_all_start_methods()[0]
        )
        if context == "fork" and (
            torch.cuda.is_initialized()
            or (dist.is_available() and dist.is_initialized())
        ):
            context = (
                "forkserver"
                if "forkserver" in multiprocessing.get_all_start_methods()
                else "spawn"
            )
        workers = filter_workers(len(dataset) // world, world, context)
    batches = scan_jobs(scan, jobs, workers, context)
    for first, stop in ranges:
        keep = np.empty(stop - first, dtype=bool)
        done = 0
        while done < len(keep):
            start, mask = next(batches)
            keep[start - first : start - first + len(mask)] = mask
            done += len(mask)
            if advance is not None:
                advance(len(mask))
        yield keep


def distributed_filter(dataset, scan, group, progress):
    """Share bit-packed masks for disjoint ranges, in dataset order."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend(group) == "nccl"
        else torch.device("cpu")
    )
    keep = np.empty(len(dataset), dtype=bool)

    def windows():
        for start in range(0, len(dataset), FILTER_ROWS_PER_RANK * world):
            n = min(len(dataset) - start, FILTER_ROWS_PER_RANK * world)
            yield [start + n * i // world for i in range(world + 1)]

    ranges = ((offsets[rank], offsets[rank + 1]) for offsets in windows())
    masks = scan_ranges(dataset, scan, ranges, world)
    for offsets in windows():
        n = offsets[-1] - offsets[0]
        size = ((n + world - 1) // world + 7) // 8
        send = np.zeros(size + 1, dtype=np.uint8)
        error = None
        try:
            mask = next(masks)
            bits = np.packbits(mask)
            send[1 : 1 + len(bits)] = bits
        except Exception as exc:
            send[0], error = 1, f"{type(exc).__name__}: {exc}"
        received = torch.empty((world, len(send)), dtype=torch.uint8, device=device)
        dist.all_gather(
            list(received.unbind(0)), torch.from_numpy(send).to(device), group=group
        )
        parts = received.cpu().numpy()
        if parts[:, 0].any():
            failed = [None] * world
            dist.all_gather_object(failed, error, group=group)
            raise ValueError(
                f"Distributed graph filter failed: {next(e for e in failed if e)}"
            )
        for bits, lo, hi in zip(parts[:, 1:], offsets, offsets[1:]):
            keep[lo:hi] = np.unpackbits(bits, count=hi - lo)
        progress.update(n)
    return keep


def filter_graphs(dataset, options, stage="train", *, graph_cutoff=None, group=None):
    """Select graphs satisfying every bound without changing cached records.

    Atom and edge counts come from the shard count index. Other filters read
    only the required degrees, geometry or labels and weights. Periodic
    distances use cached lattice shifts. ``graph_cutoff`` supplies the cutoff
    when absent from shard metadata.
    ``options`` is the result of :func:`filter_options`. If ``group`` is supplied,
    all ranks in that process group must participate with the same dataset.
    """
    if not options:
        return dataset
    if not isinstance(dataset, GraphDatasetLMDB):
        raise ValueError(
            "dataset.filter requires storage_mode='lmdb', 'clmdb' or 'memory'"
        )
    bounds = {k: v for k, v in options.items() if k not in {"max_atoms", "max_edges"}}
    fields = {_LABELS[name] for name in bounds if name in _LABELS}
    fields.update(f"{name}_weight" for name in tuple(fields))
    metadata = dataset.metadata or {}
    available = metadata.get("fields")
    cutoff = (metadata.get("graph_config") or {}).get("cutoff", graph_cutoff)
    drop_isolated = bounds.get("drop_isolated", False)
    if drop_isolated:
        degree_fields = available
        if degree_fields is None and dataset.in_memory and dataset.data_list:
            degree_fields = dataset.data_list[0]
        if degree_fields is not None and "num_neighbors" in degree_fields:
            fields.add("num_neighbors")
        else:
            fields.update(("edge_index", "edge_shifts"))
    if "min_covalent_ratio" in bounds:
        check_covalent_cutoff(dataset, bounds["min_covalent_ratio"], cutoff)
        fields.update((*_EDGE_GEOMETRY, "atomic_numbers"))
    rigid_forces = not _RIGID_FORCES.isdisjoint(bounds)
    if "drop_disconnected" in bounds or rigid_forces:
        fields.update(_EDGE_GEOMETRY)
    missing = fields.difference(available) if available is not None else ()
    if rigid_forces:
        # Unlike max_forces, rigid-motion constraints apply only to labeled
        # structures. Request optional fields even in mixed-schema shards.
        fields.update(("forces", "forces_weight"))
    if "max_forces_stress_residual" in bounds:
        fields.update(
            (
                *_EDGE_GEOMETRY,
                "pbc",
                "forces",
                "forces_weight",
                "stress",
                "stress_weight",
            )
        )
    scan = FilterScan(
        dataset, options, bounds, fields | {"edge_shifts", "lattice"}, cutoff, missing
    )
    disabled = dist.is_available() and dist.is_initialized() and dist.get_rank() != 0
    try:
        with tqdm(
            total=len(dataset), desc=f"{stage} filter", unit="graph", disable=disabled
        ) as progress:
            if fields and group is not None and dist.get_world_size(group) > 1:
                keep = distributed_filter(dataset, scan, group, progress)
            else:
                (keep,) = scan_ranges(
                    dataset, scan, [(0, len(dataset))], 1, progress.update
                )
    finally:
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
