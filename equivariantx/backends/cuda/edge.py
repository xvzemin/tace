"""CUDA evaluation of edge expressions."""

import torch

from ...ops.edge import evaluate_fake, execution_mode, input_specifications
from ...utils.metadata import parse_metadata


def evaluate(
    metadata: str,
    inputs: list[torch.Tensor],
    source: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> list[torch.Tensor]:
    from .edge_codegen import launch
    from .execution import launch as launch_tiled
    from .execution import uses_matrix_products

    if inputs[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Fused edge expressions require float32 or float64.")
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("Source and target must be one-dimensional edge indices.")
    for slot, kind, width in input_specifications(metadata):
        rows = (
            1 if kind == "shared" else source.numel() if kind == "edge" else num_nodes
        )
        if inputs[slot].numel() != rows * width:
            raise ValueError("Input shape does not match the fused expression layout.")
    inputs = [x.contiguous() for x in inputs]
    outputs = evaluate_fake(metadata, inputs, source, target, num_nodes)
    descriptions = parse_metadata(metadata)[1]
    slots = sorted({slot for _, slot, _ in descriptions})
    for slot, output in zip(slots, outputs):
        writes = [kind for _, s, kind in descriptions if s == slot]
        if writes != ["edge"] or not source.numel():
            output.zero_()
    if source.numel() and outputs:
        if uses_matrix_products(metadata):
            launch_tiled(
                metadata, inputs, source.contiguous(), target.contiguous(), outputs
            )
        else:
            launch(
                metadata,
                inputs,
                source.contiguous(),
                target.contiguous(),
                outputs,
                mode=execution_mode(metadata),
            )
    return outputs
