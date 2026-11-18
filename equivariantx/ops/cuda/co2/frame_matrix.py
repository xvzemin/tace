"""Fused Cartesian frame matrices and their mixed directional derivatives."""

from functools import lru_cache
from pathlib import Path

import torch

from ....utils.autograd import forward_ad_active
from ...common.cuda.pointwise import launch, scalar_type
from ...precision import validate_uniform
from ...torch.co2.local_frame import paired_rows


@lru_cache(maxsize=128)
def program(dtype, rank, rows, directions, transpose, batched):
    """Return one shared-memory Cartesian rotation or its matrix adjoint."""
    width = 3**rank
    threads = 128
    stages = (rank + 2) if transpose else 2
    row_bytes = (
        stages * (1 << directions) * width * (8 if dtype == torch.float64 else 4)
    )
    group = min(rows, 8, max(1, (32 << 10) // row_bytes))
    args = "".join(f"const T* d{i}, " for i in range(directions))
    names = "".join(f", d{i}" for i in range(directions))
    prefix = f"""
    using T = {scalar_type(dtype)};
    constexpr int Rank={rank}, Rows={rows}, Width={width}, Group={group};
    constexpr int Directions={directions}, Threads={threads};
    constexpr bool Batched={int(batched)};
    #define TRANSPOSE {int(transpose)}
    #define DIRECTION_ARGS {args}
    #define DIRECTION_NAMES {names}
    #define GRADIENT_ARG {"const T* gradient," if transpose else ""}
    """
    return (
        prefix + Path(__file__).with_suffix(".cu").read_text(),
        threads,
        group,
        group * row_bytes,
    )


@torch.library.custom_op(
    "eqx::cartesian_frame_matrix", mutates_args=(), device_types="cuda"
)
def contract(
    rows: torch.Tensor,
    operands: list[torch.Tensor],
    rank: int,
    transpose: bool,
) -> torch.Tensor:
    """Evaluate a rotated Cartesian row bank or its transposed frame derivative."""
    if len(operands) <= int(transpose):
        raise ValueError("Missing Cartesian rotation or cotangent.")
    validate_uniform((rows, operands))
    rotation, *directions = operands
    gradient = directions.pop() if transpose else None
    if rank not in (2, 3, 4) or rotation.shape[-2:] != (3, 3):
        raise ValueError("Cartesian frame matrices require rank 2–4 and 3×3 rotations.")
    if rows.ndim < 2 or rows.shape[-1] != 3**rank:
        raise ValueError("Incorrect Cartesian frame row width.")
    if rows.ndim != 2 and rows.shape[:-2] != rotation.shape[:-2]:
        raise ValueError("Row banks must be shared or supplied for every frame.")
    shape = (*rotation.shape[:-2], *rows.shape[-2:])
    if any(d.shape != rotation.shape for d in directions) or (
        gradient is not None and gradient.shape != shape
    ):
        raise ValueError("Incorrect Cartesian tangent or cotangent shape.")
    out_shape = rotation.shape if transpose else shape
    if not rows.numel() or not rotation.numel() or len(directions) + transpose > rank:
        return rotation.new_zeros(out_shape)
    result = rotation.new_empty(out_shape)
    source, threads, group, shared = program(
        rows.dtype, rank, rows.shape[-2], len(directions), transpose, rows.ndim != 2
    )
    frames = rotation.numel() // 9
    groups = 1 if transpose else (rows.shape[-2] + group - 1) // group
    launch(
        source,
        (*[t.contiguous() for t in (rows, *operands)], result),
        (frames,),
        frames * groups * threads,
        threads=threads,
        shared=shared,
    )
    return result


@contract.register_fake
def fake(rows, operands, rank, transpose):
    rotation = operands[0]
    shape = rotation.shape if transpose else (*rotation.shape[:-2], *rows.shape[-2:])
    return rotation.new_empty(shape)


def setup_context(ctx, inputs, output):
    rows, operands, ctx.rank, ctx.transpose = inputs
    ctx.save_for_backward(rows, *operands)


def frame_matrix(rows, rotation, directions, gradient, rank):
    operands = [rotation, *directions]
    if gradient is not None:
        operands.append(gradient)
    return contract(rows, operands, rank, gradient is not None)


def backward(ctx, cotangent):
    rows, rotation, *directions = ctx.saved_tensors
    gradient = directions.pop() if ctx.transpose else None
    rank = ctx.rank
    need_rows, needed, _, _ = ctx.needs_input_grad
    need_rotation, *need_directions = needed
    need_gradient = need_directions.pop() if ctx.transpose else False
    # A transposed map pairs one more direction with the rotation derivative.
    vectors = directions if gradient is None else [*directions, cotangent]
    covector = cotangent if gradient is None else gradient
    dr = (
        frame_matrix(rows, rotation, vectors, covector, rank) if need_rotation else None
    )
    dd = [
        frame_matrix(rows, rotation, vectors[:i] + vectors[i + 1 :], covector, rank)
        if need
        else None
        for i, need in enumerate(need_directions)
    ]
    dg = frame_matrix(rows, rotation, vectors, None, rank) if need_gradient else None
    db = None
    if need_rows:
        db = frame_matrix(
            covector,
            rotation.transpose(-1, -2),
            [v.transpose(-1, -2) for v in vectors],
            None,
            rank,
        ).sum_to_size(rows.shape)
    values = [dr, *dd]
    if ctx.transpose:
        values.append(dg)
    return db, values, None, None


contract.register_autograd(backward, setup_context=setup_context)


def evaluate(rows, rotation, rank):
    """Keep functional transforms composable; otherwise launch one native map."""
    if forward_ad_active():
        return paired_rows(rows, rotation, rank)
    return frame_matrix(rows.to(rotation), rotation, [], None, rank)
