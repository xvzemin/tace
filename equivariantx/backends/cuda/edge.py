"""CUDA evaluation of edge expressions."""

from functools import lru_cache

import torch

from ...utils.metadata import parse_metadata
from ...utils.outputs import edge_outputs


@lru_cache(maxsize=256)
def execution_mode(metadata):
    """Select element-wise execution or a narrow node reduction."""
    nodes, outputs = parse_metadata(metadata)
    if any(
        op in ("product", "matmul", "normalize", "inv_norm", "mean", "rsqrt")
        or op == "scatter"
        and nodes[args[0]][1] > 32
        for op, _, args, _ in nodes
    ):
        return "edge"
    if all(kind == "edge" for _, _, kind in outputs):
        return "pointwise"
    if len(outputs) == 1 and outputs[0][2] in ("source", "target"):
        if nodes[outputs[0][0]][1] <= 32:
            return "node"
    return "edge"


@lru_cache(maxsize=256)
def input_specifications(metadata):
    return tuple(
        (data[0], data[1], size)
        for op, size, _, data in parse_metadata(metadata)[0]
        if op == "input"
    )


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
    if any(x.device != inputs[0].device or x.dtype != inputs[0].dtype for x in inputs):
        raise ValueError("Expression inputs must have the same dtype and device.")
    if any(
        x.device != inputs[0].device or x.dtype != torch.int64 for x in (source, target)
    ):
        raise ValueError("Edge indices must be int64 tensors on the input device.")
    for slot, kind, width in input_specifications(metadata):
        rows = (
            1 if kind == "shared" else source.numel() if kind == "edge" else num_nodes
        )
        if inputs[slot].numel() != rows * width:
            raise ValueError("Input shape does not match the fused expression layout.")
    inputs = [x.contiguous() for x in inputs]
    outputs = edge_outputs(metadata, inputs, source, target, num_nodes)
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
