"""CUDA evaluation of TECE interactions."""

import torch

from .....ops.tece import base_program, contraction_fake, interaction_fake


def interaction(
    metadata: str,
    source: torch.Tensor,
    target: torch.Tensor,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    from .attention import forward

    inputs = [x.contiguous() for x in inputs]
    source, target = source.contiguous(), target.contiguous()
    base = base_program(metadata, inputs)
    for value, (width, kind) in zip(inputs, base[3]):
        rows = (
            inputs[0].shape[0]
            if kind in ("source", "target")
            else source.numel()
            if kind == "edge"
            else 1
        )
        if value.numel() != rows * width:
            raise ValueError("Input shape does not match the local interaction layout.")
    result, denominator, maximum = interaction_fake(metadata, source, target, inputs)
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
    from ...execution import launch
    from .attention import BACKWARD_TILE_SIZE

    inputs = [x.contiguous() for x in inputs]
    result = contraction_fake(metadata, source, target, inputs)
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
