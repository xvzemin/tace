"""CUDA evaluation of edge expressions."""

from functools import lru_cache

import torch

from ....utils.metadata import parse_metadata
from ....utils.outputs import edge_node_counts, edge_outputs


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


@lru_cache(maxsize=256)
def product_graph_layout(metadata):
    """Prove that padding a row-local product leaves its reductions unchanged."""
    nodes, outputs = parse_metadata(metadata)
    if not any(op == "product" for op, _, _, _ in nodes):
        return None
    kinds, zeros = {}, []
    for op, _, args, data in nodes:
        if op == "input":
            slot, kind = data
            if kind not in ("edge", "shared"):
                return None
            kinds[slot] = kind
            zero = kind == "edge"
        elif op == "constant":
            zero = data == 0
        elif op in ("add", "concat"):
            zero = all(zeros[i] for i in args)
        elif op in ("mul", "matmul"):
            zero = any(zeros[i] for i in args)
        elif op == "product":
            (_, _, paths), output, _ = data
            zero = all(
                coefficient == 0 or any(
                    role != output and index >= 0 and zeros[args[role]]
                    for role, index in enumerate(indices)
                )
                for indices, coefficient in paths
            )
        elif op in (
            "slice", "unslice", "transpose", "gather", "scatter", "silu", "tanh", "sin"
        ):
            zero = zeros[args[0]]
        elif op in ("sigmoid", "cos"):
            zero = False
        else:
            # In particular, zero padding must not introduce singular norms.
            return None
        zeros.append(zero)
    if any(
        kind not in ("edge", "shared") or (kind == "shared" and not zeros[root])
        for root, _, kind in outputs
    ):
        return None
    output_kinds = {slot: kind for _, slot, kind in outputs}
    return kinds, tuple(kind for _, kind in sorted(output_kinds.items()))


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
    from .graph import product

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
    counts = edge_node_counts(metadata, inputs, num_nodes)
    for slot, kind, width in input_specifications(metadata):
        rows = (
            1 if kind == "shared" else source.numel() if kind == "edge" else counts[kind]
        )
        if inputs[slot].numel() != rows * width:
            raise ValueError("Input shape does not match the fused expression layout.")
    inputs = [x.contiguous() for x in inputs]
    outputs = edge_outputs(metadata, inputs, source, target, num_nodes)
    descriptions = parse_metadata(metadata)[1]
    slots = sorted({slot for _, slot, _ in descriptions})
    matrix_products = uses_matrix_products(metadata)

    def run(values, results):
        values, source, target = list(values[:-2]), values[-2], values[-1]
        for slot, output in zip(slots, results):
            writes = [kind for _, s, kind in descriptions if s == slot]
            if writes != ["edge"] or not source.numel():
                output.zero_()
        if source.numel() and results:
            if matrix_products:
                launch_tiled(
                    metadata, values, source.contiguous(), target.contiguous(), results
                )
            else:
                launch(
                    metadata, values, source.contiguous(), target.contiguous(), results,
                    mode=execution_mode(metadata),
                )

    product(
        metadata, run, inputs, source, target, outputs,
        product_graph_layout(metadata) if matrix_products else None,
    )
    return outputs
