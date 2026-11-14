"""Sparse CG schedules shared by two- and three-dimensional decompositions."""

from functools import lru_cache

import torch
from torch.utils._python_dispatch import _disable_current_modes

from .. import config
from .sparse import matrix_rows


@lru_cache(maxsize=16)
def schedules(rank, scalar, vector, coupling):
    """Construct complete stages and individual paths without a dense full basis."""
    dimension = vector.dim
    transitions, groups = {}, {}
    transition_data, stages = [], []
    paths = [(scalar,)]
    plans = [()]
    for level in range(1, rank + 1):
        for path in paths:
            ir = path[-1]
            if ir in groups:
                continue
            outputs = sorted(
                vector * ir,
                key=lambda ir: ir.l if hasattr(ir, "l") else ir.m,
                reverse=True,
            )
            matrices, numbers = [], []
            for out in outputs:
                matrix = coupling(ir, out).flatten(0, 1)
                matrix = matrix / matrix.norm(dim=0)
                matrices.append(matrix)
                key = ir, out
                if key not in transitions:
                    transitions[key] = len(transition_data)
                    transition_data.append(
                        (*matrix_rows(matrix.T), *matrix_rows(matrix))
                    )
                numbers.append(transitions[key])
            matrix = torch.cat(matrices, dim=-1)
            groups[ir] = (
                tuple(outputs),
                tuple(numbers),
                (*matrix_rows(matrix.T), *matrix_rows(matrix)),
            )
        previous = dimension ** (level - 1)
        width = dimension**level
        max_forward = max(groups[p[-1]][2][0].shape[1] for p in paths)
        max_inverse = max(groups[p[-1]][2][2].shape[1] for p in paths)
        forward_index = torch.zeros(
            (width, max_forward), dtype=torch.long, device="cpu"
        )
        forward_value = torch.zeros_like(forward_index, dtype=torch.float64)
        inverse_index = torch.zeros(
            (width, max_inverse), dtype=torch.long, device="cpu"
        )
        inverse_value = torch.zeros_like(inverse_index, dtype=torch.float64)
        new_paths, new_plans = [], []
        source = destination = 0
        for path, plan in zip(paths, plans):
            ir = path[-1]
            outputs, numbers, (fi, fv, bi, bv) = groups[ir]
            selected = (
                torch.arange(dimension, device="cpu")[:, None] * previous
                + torch.arange(source, source + ir.dim, device="cpu")
            ).flatten()
            count = dimension * ir.dim
            forward_index[destination : destination + count, : fi.shape[1]] = selected[
                fi
            ]
            forward_value[destination : destination + count, : fv.shape[1]] = fv
            inverse_index[selected, : bi.shape[1]] = destination + bi
            inverse_value[selected, : bv.shape[1]] = bv
            for out, number in zip(outputs, numbers):
                new_paths.append(path + (out,))
                new_plans.append(plan + (number,))
            source += ir.dim
            destination += count
        stages.append((forward_index, forward_value, inverse_index, inverse_value))
        paths, plans = new_paths, new_plans
    return tuple(paths), tuple(plans), tuple(stages), tuple(transition_data)


def initialize(module, rank, scalar, vector, coupling, dtype, device):
    """Register immutable sparse maps; leave the compatibility matrix lazy."""
    paths, plans, stages, transitions = schedules(rank, scalar, vector, coupling)
    module.paths = paths
    # The scalar-to-vector stage is exactly the identity in both spaces.
    module.path_plans = tuple(plan[1:] for plan in plans)
    module.dimension = vector.dim
    module.lefts = tuple(vector.dim ** (rank - k - 1) for k in range(1, rank))
    sections = []
    offset = 0
    for path in paths:
        sections.append(slice(offset, offset + path[-1].dim))
        offset += path[-1].dim
    module.slices = tuple(sections)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = "cpu" if device is None else device
    module.register_buffer(
        "dtype_marker", torch.empty(0, dtype=dtype, device=device), persistent=False
    )
    module.register_buffer("change_of_basis", None, persistent=False)
    module.path_cache_names = tuple(f"path_basis_{k}" for k in range(len(paths)))
    for name in module.path_cache_names:
        module.register_buffer(name, None, persistent=False)
    module.coefficient_data = {}
    for label, data in (("stage", stages[1:]), ("transition", transitions)):
        names = []
        for k, values in enumerate(data):
            fields = []
            for suffix, value in zip(
                ("indices", "weights", "inverse_indices", "inverse_weights"), values
            ):
                name = f"{label}_{k}_{suffix}"
                fields.append(name)
                module.register_buffer(
                    name,
                    value.to(
                        dtype=dtype if value.is_floating_point() else torch.long,
                        device=device,
                        copy=True,
                    ),
                    persistent=False,
                )
                if value.is_floating_point():
                    module.coefficient_data[name] = value
            names.append(tuple(fields))
        setattr(module, label + "_names", tuple(names))
    if rank > 1 and module.dim <= config.ICTD_FUSED_MAX_WIDTH:
        indices, weights, offsets, terms = [], [], [], []
        offset = 0
        for forward_i, forward_w, reverse_i, reverse_w in stages[1:]:
            for index, weight in ((forward_i, forward_w), (reverse_i, reverse_w)):
                indices.append(index.flatten())
                weights.append(weight.flatten())
                offsets.append(offset)
                terms.append(index.shape[1])
                offset += index.numel()
        module.packed_outer = tuple(stage[0].shape[0] for stage in stages[1:])
        module.packed_offsets, module.packed_terms = tuple(offsets), tuple(terms)
        module.register_buffer(
            "packed_indices", torch.cat(indices).to(device=device), persistent=False
        )
        values = torch.cat(weights)
        module.register_buffer(
            "packed_weights", values.to(dtype=dtype, device=device), persistent=False
        )
        module.coefficient_data["packed_weights"] = values


@torch.compiler.assume_constant_result
def dense_basis(rank, dimension, dtype, device):
    """Materialize the original float64 CG basis only when explicitly needed."""
    if dimension == 2:
        from ...o2.ictd import path_matrices
    else:
        from ...o3.ictd import path_matrices
    with _disable_current_modes(), torch._C._DisableFuncTorch():
        matrix = torch.empty(
            dimension**rank, dimension**rank, dtype=torch.float64, device="cpu"
        )
        start = 0
        for _, value in path_matrices(rank):
            matrix[:, start : start + value.shape[1]] = value
            start += value.shape[1]
        return matrix.to(dtype=dtype, device=device)


@torch.compiler.assume_constant_result
def rectangular_basis(rank, dimension, index, dtype, device):
    """Build one path without allocating the complete Cartesian basis."""
    if dimension == 2:
        from ...o2.ictd import path_matrices
    else:
        from ...o3.ictd import path_matrices
    with _disable_current_modes(), torch._C._DisableFuncTorch():
        for number, (_, matrix) in enumerate(path_matrices(rank)):
            if number == index:
                return matrix.to(dtype=dtype, device=device)
    raise IndexError("Cartesian path index is out of range.")


def path_basis(module, index):
    """Cache a requested rectangular map, keeping the public dense view lazy."""
    if not hasattr(module, "path_cache_names"):
        return module.path_matrix(index)
    if index < 0:
        index += len(module.paths)
    if not 0 <= index < len(module.paths):
        raise IndexError("Cartesian path index is out of range.")
    name = module.path_cache_names[index]
    matrix = getattr(module, name)
    if matrix is None:
        marker = module.dtype_marker
        matrix = rectangular_basis(
            module.rank, module.dimension, index, marker.dtype, marker.device
        )
        setattr(module, name, matrix)
    return matrix


def restore(module, previous_dtype):
    """Regenerate floating coefficients after dtype/device changes."""
    for name, value in getattr(module, "coefficient_data", {}).items():
        setattr(module, name, value.to(getattr(module, name), copy=True))
    matrix = module._buffers.get("change_of_basis")
    if (
        matrix is not None
        and torch.finfo(matrix.dtype).eps < torch.finfo(previous_dtype).eps
    ):
        module.change_of_basis = dense_basis(
            module.rank, module.dimension, matrix.dtype, matrix.device
        )
    for index, name in enumerate(getattr(module, "path_cache_names", ())):
        matrix = module._buffers.get(name)
        if (
            matrix is not None
            and torch.finfo(matrix.dtype).eps < torch.finfo(previous_dtype).eps
        ):
            setattr(
                module,
                name,
                rectangular_basis(
                    module.rank, module.dimension, index, matrix.dtype, matrix.device
                ),
            )


def operands(module, index=None):
    """Select complete stages or the local maps belonging to one path."""
    names = (
        module.stage_names
        if index is None
        else tuple(module.transition_names[k] for k in module.path_plans[index])
    )
    return tuple(getattr(module, name) for fields in names for name in fields)
