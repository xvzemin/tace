"""CUDA evaluation of TECE interactions."""

import torch

from equivariantx.ops.common.cuda.execution import launch

from ...models.tace.tece_oam_rra.expressions import base_program
from ..tece import receiver_rows, tece_contraction_outputs, tece_outputs
from .tece_attention import BACKWARD_TILE_SIZE, forward


def interaction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    inputs = [x.contiguous() for x in inputs]
    source, target = source.contiguous(), target.contiguous()
    base = base_program(metadata, inputs)
    receivers = receiver_rows(metadata, inputs)
    for value, (width, kind) in zip(inputs, base[3]):
        if kind == "row_map":
            if value.dtype != torch.int64 or value.shape != source.shape:
                raise ValueError(
                    "Indexed radial rows require one int64 index per edge."
                )
            continue
        if isinstance(kind, tuple):
            # Each feature table owns its row count, independent of geometry.
            if value.ndim != 2 or value.shape[1] != width:
                raise ValueError("Invalid indexed radial feature table.")
            continue
        rows = (
            inputs[0].shape[0]
            if kind == "source"
            else receivers
            if kind == "target"
            else source.numel()
            if kind == "edge"
            else 1
        )
        if value.numel() != rows * width:
            raise ValueError("Input shape does not match the local interaction layout.")
    result, denominator, maximum = tece_outputs(metadata, source, target, inputs)
    if not source.numel():
        result.zero_()
        denominator.fill_(base[-1])
        maximum.zero_()
        return [result, denominator, maximum]
    forward(base, inputs, source, target, [result, denominator, maximum])
    return [result, denominator, maximum]


def contraction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate local expressions and their derivatives with CUDA kernels."""
    inputs = [x.contiguous() for x in inputs]
    result = tece_contraction_outputs(metadata, source, target, inputs)
    for value in result:
        value.zero_()
    if result and source.numel():
        launch(
            metadata,
            inputs,
            source.contiguous(),
            target.contiguous(),
            result,
            tile_size=BACKWARD_TILE_SIZE,
        )
    return result
