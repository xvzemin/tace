################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Neighbor-list construction and backend behavior.

The single-system interface returns the physical lattice unchanged. The packed
batch interface returns NumPy arrays and builds all AlchemiOps systems together,
including systems with different cells and periodic boundary conditions.

Missing cell vectors are completed only for backend computation; the physical
lattice returned to the model is unchanged. Shift components along nonperiodic
directions are always normalized to zero, and only zero-shift self edges are
removed.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from ase.geometry import complete_cell
from matscipy.neighbours import neighbour_list

try:
    from ase.neighborlist import primitive_neighbor_list
except ImportError:
    pass
try:
    from vesin import NeighborList as vesin_nl
except ImportError:
    pass


NL_BACKEND = ["ase", "matscipy", "vesin", "alchemiops"]

# For alchemiops
NV_CELL_LIST_THRESHOLD = 1024
NV_NONPERIODIC_CELL_LIST_THRESHOLD = 4096
NV_CPU_CELL_LIST_THRESHOLD = 128


def grow_search_capacity(capacity: int) -> int:
    return max(capacity + 1, (capacity * 5 + 3) // 4)


def choose_nv_method(n_atoms: int, periodic: bool, device) -> str:
    if device.type == "cpu":
        threshold = NV_CPU_CELL_LIST_THRESHOLD
    elif periodic:
        threshold = NV_CELL_LIST_THRESHOLD
    else:
        threshold = NV_NONPERIODIC_CELL_LIST_THRESHOLD
    return "batch_cell_list" if n_atoms >= threshold else "batch_naive"


def get_nv_start_capacity(n_atoms: int, max_neighbors: Union[int, str, None]) -> int:
    # A periodic atom can have many images of the same neighbor, even for N=1.
    # This is an allocation hint, never a cap on the returned neighbor count.
    if max_neighbors is None or max_neighbors == "inf":
        return 128
    return max(1, int(max_neighbors))


def build_alchemiops_batch(
    positions: np.ndarray,
    ptr: np.ndarray,
    cutoff: float,
    pbc: np.ndarray,
    lattice: np.ndarray,
    device=None,
):
    try:
        import torch
        from nvalchemiops.torch.neighbors import neighbor_list as nvidia_neighbor_list
    except ImportError as exc:
        raise ImportError(
            "neighborlist_backend='alchemiops' requires "
            "`nvalchemi-toolkit-ops` from https://github.com/NVIDIA/nvalchemi-toolkit-ops."
        ) from exc

    device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type not in {"cpu", "cuda"}:
        raise NotImplementedError(f"AlchemiOps does not support device {device}")
    dtype = torch.float64
    num_atoms = int(positions.shape[0])
    if num_atoms > np.iinfo(np.int32).max or len(ptr) - 1 > np.iinfo(np.int32).max:
        raise NotImplementedError("AlchemiOps batch indices must fit in int32")
    counts = np.diff(ptr)
    periodic = bool(pbc.any())

    pos = torch.as_tensor(positions, dtype=dtype, device=device)
    if periodic:
        # Nonperiodic systems need an invertible temporary cell when mixed with
        # periodic systems. Their physical cells are never changed or returned.
        neighbor_cells = np.stack(
            [
                complete_cell(c) if flags.any() else np.eye(3)
                for c, flags in zip(lattice, pbc)
            ]
        )
        if not np.isfinite(neighbor_cells).all() or np.any(
            np.linalg.det(neighbor_cells) == 0
        ):
            raise NotImplementedError("AlchemiOps requires nonsingular search cells")
        cell = torch.as_tensor(neighbor_cells, dtype=dtype, device=device)
        pbc_tensor = torch.as_tensor(pbc, dtype=torch.bool, device=device)
    else:
        cell = None
        pbc_tensor = None

    batch_idx = torch.as_tensor(
        np.repeat(np.arange(len(counts), dtype=np.int32), counts), device=device
    )
    batch_ptr = torch.as_tensor(ptr, dtype=torch.int32, device=device)
    max_atoms = int(counts.max())
    method = choose_nv_method(max_atoms, periodic=periodic, device=device)
    if periodic:
        # Naive explicitly enumerates all images, supports partial PBC, and
        # corrects shifts for unwrapped atoms. Do not use cluster_tile (FP32).
        # Keep cell-list selection to its ordinary fully periodic, positive-cell
        # domain; mixed PBC and multiple images use the general naive path.
        heights = 1.0 / np.linalg.norm(np.linalg.inv(neighbor_cells), axis=1)
        if (
            not pbc.all()
            or np.any(np.linalg.det(neighbor_cells) <= 0)
            or np.any(2 * cutoff >= heights)
        ):
            method = "batch_naive"
    extra_kwargs = {}
    if method == "batch_naive":
        extra_kwargs["max_atoms_per_system"] = max_atoms

    search_capacity = get_nv_start_capacity(max_atoms, None)
    while True:
        result = nvidia_neighbor_list(
            pos,
            float(cutoff),
            cell=cell,
            pbc=pbc_tensor,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            method=method,
            max_neighbors=int(search_capacity),
            return_neighbor_list=False,
            **extra_kwargs,
        )
        if len(result) == 2:
            neighbor_matrix, num_neighbors = result
            shifts = torch.zeros(
                (*neighbor_matrix.shape, 3), dtype=torch.int32, device=device
            )
        else:
            neighbor_matrix, num_neighbors, shifts = result

        max_found = int(num_neighbors.max().item()) if num_neighbors.numel() else 0
        if max_found <= search_capacity:
            break
        search_capacity = max(max_found, grow_search_capacity(search_capacity))

    total_atoms, capacity = neighbor_matrix.shape
    slots = torch.arange(capacity, dtype=torch.long, device=device).expand(
        total_atoms, capacity
    )
    valid = (slots < num_neighbors.unsqueeze(1)).reshape(-1)
    edge_slots = torch.nonzero(valid, as_tuple=False).flatten()

    if edge_slots.numel() == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 3), dtype=np.int64)

    center = (edge_slots // capacity).to(dtype=torch.long)
    neighbor = neighbor_matrix.reshape(-1).index_select(0, edge_slots).to(torch.long)
    shifts = shifts.reshape(-1, 3).index_select(0, edge_slots).to(torch.long)

    # Transfer the complete chunk once instead of each graph or tensor. Rows
    # remain source-sorted, so edges belonging to each graph are contiguous.
    packed = (
        torch.cat((center[:, None], neighbor[:, None], shifts), dim=1).cpu().numpy()
    )
    edge_index = np.ascontiguousarray(packed[:, :2].T)
    edge_shifts = np.ascontiguousarray(packed[:, 2:])
    if np.any(edge_index < 0) or np.any(edge_index >= num_atoms):
        raise RuntimeError("AlchemiOps returned an invalid atom index")
    graph_indices = np.searchsorted(ptr, edge_index, side="right") - 1
    if np.any(graph_indices[0] != graph_indices[1]):
        raise RuntimeError("AlchemiOps returned a cross-system edge")
    edge_shifts[~pbc[graph_indices[0]]] = 0
    keep = (edge_index[0] != edge_index[1]) | np.any(edge_shifts != 0, axis=1)
    return edge_index[:, keep], edge_shifts[keep]


def as_numpy(value):
    """Accept NumPy-like values and detached CPU/CUDA Torch tensors."""
    if hasattr(value, "detach"):
        value = value.detach()
        if value.is_floating_point():
            value = value.double()
        value = value.cpu().numpy()
    return np.asarray(value)


def get_neighborhood_batch(
    positions,
    ptr,
    cutoff: float,
    pbc=None,
    lattice=None,
    max_neighbors: Optional[int] = None,
    backend: str = "matscipy",
    device=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build complete directed neighbor lists for a packed batch.

    Inputs may be NumPy arrays or Torch tensors. ``positions`` has shape (N, 3)
    and integer ``ptr`` has shape (B + 1,), starting at zero and ending at N.
    Atoms of each system must be contiguous; empty systems are allowed. ``pbc``
    is a shared bool/(3,) array or per-system (B, 3) flags. ``lattice`` is a
    shared (3, 3) or per-system (B, 3, 3) row-vector cell, defaulting to zero.

    Return NumPy int64 arrays ``edge_index`` (2, E), ``edge_shifts`` (E, 3), and
    ``edge_ptr`` (B + 1,). Atom indices are global; graph b occupies edge slice
    ``edge_ptr[b]:edge_ptr[b + 1]``. Shift vectors satisfy
    ``r_ij = positions[j] - positions[i] + edge_shifts @ lattice[b]``.
    Positions and search cells use FP64, physical input cells are not modified,
    and nonzero-shift self images are retained. ``max_neighbors`` is accepted
    for compatibility but never truncates edges.

    CPU backends use the single-system implementation. AlchemiOps operates on
    the whole chunk on ``device`` (default CUDA when available, otherwise CPU),
    expanding its allocation and repeating the complete search on overflow.
    Unsupported backend combinations warn and fall back to Matscipy; missing
    dependencies and unexpected runtime failures are not silently suppressed.
    """
    if backend not in NL_BACKEND:
        raise ValueError(
            f"Unknown neighborlist backend {backend!r}; expected {NL_BACKEND}"
        )
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    positions = np.ascontiguousarray(as_numpy(positions), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("Positions must contain finite values")
    ptr = as_numpy(ptr)
    if ptr.ndim != 1 or not ptr.size or not np.issubdtype(ptr.dtype, np.integer):
        raise ValueError("ptr must be a nonempty one-dimensional integer array")
    ptr = ptr.astype(np.int64, copy=False)
    if ptr[0] != 0 or ptr[-1] != len(positions) or np.any(np.diff(ptr) < 0):
        raise ValueError("ptr must start at zero, be nondecreasing, and end at N")
    num_graphs = len(ptr) - 1
    pbc = np.asarray(False if pbc is None else as_numpy(pbc), dtype=bool)
    if pbc.shape not in {(), (3,), (num_graphs, 3)}:
        raise ValueError("pbc must be a bool, shape (3,), or shape (B, 3)")
    pbc = np.broadcast_to(pbc, (num_graphs, 3)).copy()
    if lattice is None:
        lattice = np.zeros((num_graphs, 3, 3), dtype=np.float64)
    else:
        lattice = np.asarray(as_numpy(lattice), dtype=np.float64)
        if lattice.shape not in {(3, 3), (num_graphs, 3, 3)}:
            raise ValueError("lattice must have shape (3, 3) or (B, 3, 3)")
        lattice = np.broadcast_to(lattice, (num_graphs, 3, 3)).copy()
    if not np.isfinite(lattice).all():
        raise ValueError("The cell must contain finite values")
    if np.any(pbc.any(axis=1) & np.all(np.isclose(lattice, 0), axis=(1, 2))):
        raise ValueError("A periodic structure requires a nonzero lattice")
    if np.any(pbc.all(axis=1) & (np.linalg.det(lattice) == 0)):
        raise ValueError("A fully periodic structure requires a nonsingular cell")
    edge_ptr = np.zeros(num_graphs + 1, dtype=np.int64)
    if not len(positions):
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            edge_ptr,
        )

    if backend == "alchemiops":
        try:
            edge_index, shifts = build_alchemiops_batch(
                positions, ptr, cutoff, pbc, lattice, device=device
            )
        except NotImplementedError as exc:
            warnings.warn(
                f"AlchemiOps batch unsupported ({exc}); falling back to Matscipy",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = "matscipy"
        else:
            edge_ptr = np.searchsorted(edge_index[0], ptr).astype(np.int64)
            return edge_index, shifts, edge_ptr

    indices, all_shifts = [], []
    for graph, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        if stop > start:
            edges, shifts, _, _ = get_neighborhood(
                positions[start:stop],
                cutoff,
                tuple(pbc[graph]),
                lattice[graph],
                max_neighbors=max_neighbors,
                backend=backend,
            )
            indices.append(edges + start)
            all_shifts.append(shifts)
            edge_ptr[graph + 1] = edge_ptr[graph] + edges.shape[1]
        else:
            edge_ptr[graph + 1] = edge_ptr[graph]
    return (
        np.concatenate(indices, axis=1).astype(np.int64, copy=False),
        np.concatenate(all_shifts, axis=0).astype(np.int64, copy=False),
        edge_ptr,
    )


def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: Union[bool, Tuple[bool, bool, bool], None] = None,
    lattice: Union[np.ndarray, None] = None,  # [3, 3]
    max_neighbors: Union[int, None] = None,
    backend: str = "matscipy",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[bool, bool, bool],
    np.ndarray,
]:
    positions = np.asarray(positions, dtype=np.float64)
    if backend not in NL_BACKEND:
        raise ValueError(
            f"Unknown neighborlist backend '{backend}'. "
            f"Supported backends: {NL_BACKEND}"
        )

    # === PBC ===
    if pbc is None:
        pbc = (False, False, False)
    elif isinstance(pbc, bool):
        pbc = (pbc,) * 3
    else:
        pbc = tuple(bool(i) for i in pbc)
    if len(pbc) != 3:
        raise ValueError(f"pbc must contain three values, got {pbc}")

    # Keep the physical lattice unchanged and construct a complete temporary
    # cell only for neighbor-list backends that require a nonsingular cell.
    if lattice is None:
        lattice = np.zeros((3, 3), dtype=positions.dtype)
    else:
        lattice = np.asarray(lattice, dtype=positions.dtype)
        if lattice.shape != (3, 3):
            raise ValueError(f"lattice must have shape (3, 3), got {lattice.shape}")
        lattice = lattice.copy()
    if any(pbc) and np.allclose(lattice, 0.0):
        raise ValueError(
            "At least one direction is periodic, but lattice is None or zero."
        )
    neighbor_cell = complete_cell(lattice)

    # === Neighborlist ===
    if backend == "matscipy":
        edges = neighbour_list(
            quantities="ijSd",
            pbc=pbc,
            cell=neighbor_cell,
            positions=positions,
            cutoff=cutoff,
        )
    elif backend == "vesin":
        # https://github.com/Luthaf/vesin/blob/main/python/vesin/vesin/_ase.py
        edges = vesin_nl(cutoff=cutoff, full_list=True).compute(
            points=positions,
            box=neighbor_cell,
            periodic=pbc,
            quantities="ijSd",
        )
        edges = list(edges)
        edges[0] = edges[0].astype(np.int64)
        edges[1] = edges[1].astype(np.int64)
        edges = tuple(edges)
    elif backend == "ase":
        edges = primitive_neighbor_list(
            "ijSd",
            pbc,
            neighbor_cell,
            positions,
            cutoff=cutoff,
            self_interaction=False,
            use_scaled_positions=False,
        )
    elif backend == "alchemiops":
        edge_index, edge_shifts, _ = get_neighborhood_batch(
            positions,
            np.array([0, len(positions)]),
            cutoff,
            pbc=pbc,
            lattice=neighbor_cell,
            backend="alchemiops",
        )
        edges = edge_index[0], edge_index[1], edge_shifts

    # max_neighbors is retained in model/config interfaces for compatibility,
    # but neighbor truncation is intentionally disabled.
    source, target, shifts = edges[:3]

    real_self_loop = source == target
    real_self_loop &= np.all(shifts == 0, axis=1)
    keep_edge = ~real_self_loop

    source = source[keep_edge]
    target = target[keep_edge]

    edge_shifts = shifts[keep_edge]
    # Matscipy can report temporary-cell offsets along nonperiodic directions.
    edge_shifts[:, np.logical_not(pbc)] = 0
    edge_index = np.stack((source, target))

    return edge_index, edge_shifts, pbc, lattice


def get_neighborhood_for_calculator(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    lattice: Optional[np.ndarray] = None,  # [3, 3]
    max_neighbors: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    edge_index, edge_shifts, _, _ = get_neighborhood(
        positions=positions,
        cutoff=cutoff,
        pbc=pbc,
        lattice=lattice,
        max_neighbors=max_neighbors,
        backend="matscipy",
    )
    return edge_index, edge_shifts
