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
def row_index_slots(metadata):
    return {
        k[1] for _, k, _ in input_specifications(metadata) if isinstance(k, tuple)
    } | {k[1] for _, _, k in parse_metadata(metadata)[1] if isinstance(k, tuple)}


@lru_cache(maxsize=256)
def padding_layout(metadata, indexed=False):
    """Prove that padded rows cannot contribute to any shared output."""
    nodes, outputs = parse_metadata(metadata)
    operations = {op for op, *_ in nodes}
    required = {"frame", "matmul"} if indexed else {"product"}
    if not required <= operations:
        return None
    kinds, zeros = {}, []

    def merge(mapping, slot, kind):
        previous = mapping.get(slot)
        if previous is None or previous == kind:
            mapping[slot] = kind
            return True
        if indexed and {previous, kind} <= {"source", "target", "both"}:
            mapping[slot] = "both"
            return True
        return False

    for op, _, args, data in nodes:
        if op == "input":
            slot, kind = data
            if isinstance(kind, tuple):
                return None
            if not indexed and kind not in ("edge", "shared"):
                return None
            if not merge(kinds, slot, kind):
                return None
            zero = kind != "shared"
        elif op == "constant":
            zero = data == 0
        elif op in ("add", "concat"):
            zero = all(zeros[i] for i in args)
        elif op in ("mul", "matmul"):
            zero = any(zeros[i] for i in args)
        elif op == "frame":
            zero = any(zeros[j] for role, j in enumerate(args) if role != data[1])
        elif op == "product":
            (_, _, paths), output, _ = data
            zero = all(
                coefficient == 0
                or any(
                    role != output and index >= 0 and zeros[args[role]]
                    for role, index in enumerate(indices)
                )
                for indices, coefficient in paths
            )
        elif op in (
            "slice",
            "unslice",
            "transpose",
            "gather",
            "scatter",
            "silu",
            "tanh",
            "sin",
        ):
            zero = zeros[args[0]]
        elif op in ("sigmoid", "cos"):
            zero = False
        elif op == "activation_vjp":
            zero = zeros[args[1]]
        else:
            # In particular, zero padding must not introduce singular norms.
            return None
        zeros.append(zero)
    output_kinds = {}
    for root, slot, kind in outputs:
        if (
            isinstance(kind, tuple)
            or (not indexed and kind not in ("edge", "shared"))
            or (kind == "shared" and not zeros[root])
            or not merge(output_kinds, slot, kind)
        ):
            return None
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
    from .graph import edge_program

    if inputs[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Fused edge expressions require float32 or float64.")
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("Source and target must be one-dimensional edge indices.")
    if any(
        x.device != inputs[0].device or x.dtype not in (inputs[0].dtype, torch.int64)
        for x in inputs
    ):
        raise ValueError("Expression inputs must have the same dtype and device.")
    if any(
        x.device != inputs[0].device or x.dtype != torch.int64 for x in (source, target)
    ):
        raise ValueError("Edge indices must be int64 tensors on the input device.")
    counts = edge_node_counts(metadata, inputs, num_nodes)
    descriptions = parse_metadata(metadata)[1]
    for slot in row_index_slots(metadata):
        if inputs[slot].dtype != torch.int64 or inputs[slot].shape != source.shape:
            raise ValueError("Each row index must supply one int64 index per edge.")
    for slot, kind, width in input_specifications(metadata):
        rows = (
            1
            if kind == "shared"
            else source.numel()
            if kind == "edge"
            else inputs[kind[2]].shape[0]
            if isinstance(kind, tuple)
            else counts[kind]
        )
        if (
            inputs[slot].dtype != inputs[0].dtype
            or inputs[slot].numel() != rows * width
        ):
            raise ValueError("Input shape does not match the fused expression layout.")
    inputs = [x.contiguous() for x in inputs]
    outputs = edge_outputs(metadata, inputs, source, target, num_nodes)
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
                    metadata,
                    values,
                    source.contiguous(),
                    target.contiguous(),
                    results,
                    mode=execution_mode(metadata),
                )

    edge_program(
        metadata,
        run,
        inputs,
        source,
        target,
        outputs,
        padding_layout(metadata) if matrix_products else None,
        counts,
    )
    return outputs
