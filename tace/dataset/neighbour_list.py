################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Neighbor-list construction and backend behavior.

``test/test_neighbour_list.py`` covers five system
classes for each of ASE, Matscipy, Vesin, and AlchemiOps: nonperiodic without a
cell, nonperiodic with a cell, 1D periodic, 2D periodic, and 3D periodic.

Backend-specific handling:

* Matscipy requires a complete, nonsingular cell.
* AlchemiOps may add batch processing, in a future update.

Missing cell vectors are completed only for backend computation; the physical
lattice returned to the model is unchanged. Shift components along nonperiodic
directions are always normalized to zero, and only zero-shift self edges are
removed.
"""

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

# # Disable max_neighbors
# def filter_max_neighbors(source, target, shifts, distances, max_neighbors="inf"):

#     if max_neighbors is None or max_neighbors == "inf":
#         return source, target, shifts
#     order = np.lexsort((distances, source))
#     src_sorted = source[order]
#     dst_sorted = target[order]
#     shifts_sorted = shifts[order]

#     unique_src, counts = np.unique(src_sorted, return_counts=True)
#     cum_counts = np.cumsum(counts)  # [3, 2, 1] => [3, 5, 6]

#     mask = np.zeros(len(src_sorted), dtype=bool)
#     start_idx = 0
#     for end_idx in cum_counts:
#         count = end_idx - start_idx
#         keep = min(max_neighbors, count)
#         mask[start_idx : start_idx + keep] = True
#         start_idx = end_idx

#     return (
#         src_sorted[mask],
#         dst_sorted[mask],
#         shifts_sorted[mask],
#     )


def _grow_search_capacity(capacity: int) -> int:
    return max(capacity + 1, (capacity * 5 + 3) // 4)


def _choose_nv_method(n_atoms: int, periodic: bool, device) -> str:
    if device.type == "cpu":
        threshold = NV_CPU_CELL_LIST_THRESHOLD
    elif periodic:
        threshold = NV_CELL_LIST_THRESHOLD
    else:
        threshold = NV_NONPERIODIC_CELL_LIST_THRESHOLD
    return "batch_cell_list" if n_atoms >= threshold else "batch_naive"


def _get_nv_start_capacity(n_atoms: int, max_neighbors: Union[int, str, None]) -> int:
    if max_neighbors is None or max_neighbors == "inf":
        return max(1, min(max(n_atoms - 1, 1), 64))
    return max(1, int(max_neighbors))


def _build_alchemiops_edges(
    positions: np.ndarray,
    cutoff: float,
    pbc: Tuple[bool, bool, bool],
    lattice: Union[np.ndarray, None],
    max_neighbors: Union[int, str, None],
):
    try:
        import torch
        from nvalchemiops.torch.neighbors import neighbor_list as nvidia_neighbor_list
    except ImportError as exc:
        raise ImportError(
            "neighborlist_backend='alchemiops' requires "
            "`nvalchemi-toolkit-ops` from https://github.com/NVIDIA/nvalchemi-toolkit-ops."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float64 if np.asarray(positions).dtype == np.float64 else torch.float32
    )
    num_atoms = int(positions.shape[0])
    periodic = any(pbc)

    pos = torch.as_tensor(positions, dtype=dtype, device=device)
    if periodic:
        cell = (
            torch.as_tensor(lattice, dtype=dtype, device=device)
            .reshape(1, 3, 3)
            .contiguous()
        )
        pbc_tensor = torch.tensor([pbc], dtype=torch.bool, device=device)
    else:
        cell = None
        pbc_tensor = None

    batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=device)
    batch_ptr = torch.tensor([0, num_atoms], dtype=torch.int32, device=device)
    method = _choose_nv_method(num_atoms, periodic=periodic, device=device)
    extra_kwargs = {}
    if method == "batch_naive":
        extra_kwargs["max_atoms_per_system"] = num_atoms

    search_capacity = _get_nv_start_capacity(num_atoms, max_neighbors)
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
        search_capacity = max(max_found, _grow_search_capacity(search_capacity))

    total_atoms, capacity = neighbor_matrix.shape
    slots = torch.arange(capacity, dtype=torch.long, device=device).expand(
        total_atoms, capacity
    )
    valid = (slots < num_neighbors.unsqueeze(1)).reshape(-1)
    edge_slots = torch.nonzero(valid, as_tuple=False).flatten()

    if edge_slots.numel() == 0:
        empty_index = np.empty((0,), dtype=np.int64)
        empty_shifts = np.empty((0, 3), dtype=np.int64)
        empty_distances = np.empty((0,), dtype=np.float64)
        return empty_index, empty_index, empty_shifts, empty_distances

    center = (edge_slots // capacity).to(dtype=torch.long)
    neighbor = neighbor_matrix.reshape(-1).index_select(0, edge_slots).to(torch.long)
    shifts = shifts.reshape(-1, 3).index_select(0, edge_slots).to(torch.long)

    source = center
    target = neighbor
    edge_vec = pos.index_select(0, target) - pos.index_select(0, source)
    if periodic:
        edge_vec = edge_vec + shifts.to(dtype=dtype) @ cell[0]
    distances = torch.linalg.vector_norm(edge_vec, dim=1)

    return (
        source.cpu().numpy().astype(np.int64),
        target.cpu().numpy().astype(np.int64),
        shifts.cpu().numpy().astype(np.int64),
        distances.cpu().numpy(),
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
        edges = _build_alchemiops_edges(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            lattice=neighbor_cell,
            max_neighbors=None,
        )

    # max_neighbors is retained in model/config interfaces for compatibility,
    # but neighbor truncation is intentionally disabled.
    # source, target, shifts = filter_max_neighbors(
    #     *edges, max_neighbors=max_neighbors
    # )
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
