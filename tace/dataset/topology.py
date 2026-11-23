################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

try:
    from ._connectivity import bulk as native_bulk
except ImportError:
    native_bulk = None


def adjacency(source, target, size):
    return csr_matrix(
        (np.ones(len(source), dtype=bool), (source, target)), shape=(size, size)
    )


def component_images(source, target, shifts, n, roots):
    """Unwrap a spanning forest into integer cell images."""
    images = np.zeros((n, 3), dtype=np.int64)
    if not shifts.any():
        return images
    # A fictitious root starts one tree per component in a single traversal.
    graph = adjacency(
        np.r_[source, np.full(len(roots), n)], np.r_[target, roots], n + 1
    )
    _, parent = breadth_first_order(graph, n, directed=False)
    parent = parent[:n]
    parent[roots] = roots
    for i, j, sign in ((source, target, 1), (target, source, -1)):
        tree = (parent[j] == i) & (i != j)
        images[j[tree]] = sign * shifts[tree]
    while not np.array_equal(parent[parent], parent):
        images += images[parent]
        parent = parent[parent]
    return images


def winding_basis(cycles):
    """Return independent integer winding vectors without a floating-point rank test."""
    nz = np.flatnonzero(cycles.any(axis=1))
    if not nz.size:
        return cycles[:0]
    u = cycles[nz[0]]
    normals = np.cross(u, cycles)
    nz2 = np.flatnonzero(normals.any(axis=1))
    if not nz2.size:
        return u[None]
    j = nz2[0]
    nz3 = np.flatnonzero(cycles @ normals[j])
    idx = [nz[0], j, nz3[0]] if nz3.size else [nz[0], j]
    return cycles[idx]


def forces_stress_residual(x, forces, stress, cell, labels, images, source, cycles):
    """Frobenius norm of the constrained stress residual, in eV/angstrom³.

    A component winding in the subspace W has free symmetric stress only
    within W. Column directions perpendicular to W are fixed by its force
    virial. Sum components before projecting out their free stresses.
    """
    volume = abs(np.linalg.det(cell))
    if not volume > 0:
        return 0.0  # No volume-normalized cell stress is defined.
    residual = np.asarray(stress, dtype=np.float64).reshape(3, 3).copy()
    free = []
    for c in range(labels.max() + 1):
        basis = winding_basis(cycles[labels[source] == c])
        if len(basis) == 3:
            return 0.0  # This component can supply any symmetric cell stress.
        ids = np.flatnonzero(labels == c)
        y = (x[ids] - x[ids[0]]) + (images[ids] - images[ids[0]]) @ cell
        f = forces[ids]
        b = -(f - f.mean(axis=0)).T @ (y - y.mean(axis=0)) / volume
        if not len(basis):
            residual -= (b + b.T) / 2
            continue
        q = np.linalg.qr((basis @ cell).T, mode="reduced")[0].T
        w = q.T @ q
        h = np.eye(3) - w
        cross = w @ b @ h
        residual -= h @ ((b + b.T) / 2) @ h + cross + cross.T
        for j, u in enumerate(q):
            for v in q[j:]:
                a = np.outer(u, v)
                free.append((a + a.T).ravel() / 2)
    v = residual.ravel()
    if free:
        basis = np.stack(free, axis=1)
        coeff, _, rank, _ = np.linalg.lstsq(
            basis, v, rcond=8 * max(6, len(free)) * np.finfo(float).eps
        )
        if rank == 6:
            return 0.0
        v = v - basis @ coeff
    return np.linalg.norm(v)


def rotation_forces_square(x, f):
    """Squared force projection onto rigid rotations of one finite component."""
    if len(x) == 2:
        length = np.linalg.norm(x[1] - x[0])
        if length == 0:
            return 0.0
        axis = (x[1] - x[0]) / length
        delta = f[1] - f[0]
        transverse = delta - (delta @ axis) * axis
        return 0.5 * (transverse @ transverse)
    x = x - x.mean(axis=0)
    rotations = np.cross(np.eye(3)[:, None, :], x).reshape(3, -1).T
    u, s, _ = np.linalg.svd(rotations, full_matrices=False)
    rank = s > s[0] * rotations.shape[0] * np.finfo(np.float64).eps
    projection = u[:, rank].T @ (f - f.mean(axis=0)).ravel()
    return projection @ projection


def bulk_certificate(source, target, shifts, x, cell, cutoff):
    """Prove connectivity and three independent winding cycles using at most N+2 edges."""
    # A zero-shift edge to a smaller site for each site except zero is a
    # spanning tree. Any shifted edge then closes a cycle with its own shift.
    a, b, c = (axis == 0 for axis in shifts.T)
    same = a & b & c
    witnesses = []
    for axis in (~a & b & c, a & ~b & c, a & b & ~c):
        if not axis.any():
            return False
        witnesses.append(axis.argmax())
    tree_edges = np.flatnonzero(same & (source != target))
    child = np.maximum(source[tree_edges], target[tree_edges])
    tree = np.full(len(x), -1, dtype=np.intp)
    tree[child] = tree_edges
    if (tree[1:] < 0).any():
        return False
    if cutoff is None:
        return True
    ids = np.concatenate((tree[1:], witnesses))
    i, j, shift = source[ids], target[ids], shifts[ids]
    vector = x[j] - x[i] + shift @ cell
    square = np.einsum("ij,ij->i", vector, vector)
    # Only certify strict interior edges; the full path decides cutoff ties.
    return bool((square < np.nextafter(cutoff**2, 0.0)).all())


def passes_topology(
    graph,
    n,
    cutoff,
    drop_disconnected,
    max_rotation_forces_rmse,
    max_translation_forces_rmse,
    max_forces_stress_residual=None,
):
    """Check enabled topology predicates on the active cutoff graph."""
    if n == 0:
        return True
    f = graph.get("forces")
    weight = graph.get("forces_weight")
    check_forces = (
        f is not None and weight is not None and np.all(np.asarray(weight) > 0)
    )
    translate = check_forces and max_translation_forces_rmse is not None
    rotate = check_forces and max_rotation_forces_rmse is not None
    virial = (
        max_forces_stress_residual is not None
        and check_forces
        and graph.get("stress") is not None
        and graph.get("stress_weight") is not None
        and np.all(np.asarray(graph["stress_weight"]) > 0)
        and graph.get("pbc") is not None
        and np.asarray(graph["pbc"]).any()
    )
    if translate:
        f = np.asarray(f, dtype=np.float64)
        # Exact for a connected graph; a lower bound even for disconnected ones.
        whole = np.linalg.norm(f.mean(axis=0)) / np.sqrt(3.0)
        if not whole <= max_translation_forces_rmse:
            return False
    if (n == 1 and not virial) or not (
        drop_disconnected or translate or rotate or virial
    ):
        return True
    edge_index = np.asarray(graph["edge_index"])
    source, target = edge_index
    shifts = np.asarray(graph["edge_shifts"], dtype=np.int64)
    x = np.asarray(graph["positions"], dtype=np.float64)
    cell = np.asarray(graph["lattice"], dtype=np.float64).reshape(3, 3)
    if virial and len(source):
        if native_bulk is None:
            certified = bulk_certificate(source, target, shifts, x, cell, cutoff)
        else:
            certified = native_bulk(
                edge_index, shifts, x, cell, -1.0 if cutoff is None else cutoff**2
            )
        if certified:
            return True
    if cutoff is not None:
        v = x[target] - x[source] + shifts @ cell
        inside = np.einsum("ei,ei->e", v, v) < cutoff**2
        if not inside.all():
            source, target, shifts = source[inside], target[inside], shifts[inside]
    if (rotate or virial) and shifts.any():
        # If zero-shift edges already connect every site, any shifted edge
        # closes a winding cycle. No unwrapping or rotation projection is needed.
        zero = ~shifts.any(axis=1)
        if (
            np.count_nonzero(zero) >= n - 1
            and connected_components(
                adjacency(source[zero], target[zero], n),
                directed=False,
                return_labels=False,
            )
            == 1
            and (not virial or len(winding_basis(shifts[~zero])) == 3)
        ):
            return True
    count, labels = connected_components(adjacency(source, target, n), directed=False)
    if drop_disconnected and count > 1:
        return False
    if translate and count > 1:
        # The minimum force correction subtracts each component's mean force.
        sizes = np.bincount(labels)
        sums = np.stack([np.bincount(labels, weights=col) for col in f.T])
        rmse = np.sqrt(np.sum(sums**2 / sizes) / (3 * n))
        if not rmse <= max_translation_forces_rmse:
            return False
    if not virial and (not rotate or count == n):
        return True
    # A nonzero self-image is already a certificate of periodic winding.
    if count == 1 and shifts[source == target].any() and not virial:
        return True
    _, roots, sizes = np.unique(labels, return_index=True, return_counts=True)
    images = component_images(source, target, shifts, n, roots)
    cycles = shifts + images[source] - images[target]
    if virial:
        f = np.asarray(f, dtype=np.float64)
        residual = forces_stress_residual(
            x, f, graph["stress"], cell, labels, images, source, cycles
        )
        if not residual <= max_forces_stress_residual:
            return False
    if not rotate or count == n:
        return True
    winding = np.zeros(count, dtype=bool)
    winding[labels[source[cycles.any(axis=1)]]] = True
    finite = (~winding & (sizes > 1))[labels]
    indices = np.flatnonzero(finite)
    if not indices.size:
        return True
    f = np.asarray(f, dtype=np.float64)
    if not np.isfinite(f[indices]).all():
        return False
    indices = indices[np.argsort(labels[indices], kind="stable")]
    square = 0.0
    for ids in np.split(indices, np.flatnonzero(np.diff(labels[indices])) + 1):
        y = (x[ids] - x[ids[0]]) + (images[ids] - images[ids[0]]) @ cell
        square += rotation_forces_square(y, f[ids])
    return np.sqrt(square / (3 * n)) <= max_rotation_forces_rmse
