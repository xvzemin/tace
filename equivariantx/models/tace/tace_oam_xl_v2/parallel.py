"""Overlap independent parameter GEMMs with the explicit edge adjoint sweep.

This module rewrites already differentiated graphs only. A paired operator
forks and rejoins CUDA streams within one call, so Inductor sees its ordinary
input/output lifetime contract and can keep its existing memory planner.
"""

from functools import lru_cache
import operator
import os
from contextlib import nullcontext

import torch
from .lt_matrix import selected_mm

from equivariantx.utils.outputs import o3_outputs
from .edge import scheduled_edges


@lru_cache(maxsize=16)
def matrix_stream(device, parent_stream=0):
    """Own one auxiliary stream per device and keep it alive through replay."""
    return torch.cuda.Stream(device=device)


@torch.library.custom_op(
    "eqx_oam_xl::parallel_edge_mm", mutates_args=(), device_types="cuda"
)
def parallel_edge_mm(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
    left: list[torch.Tensor],
    right: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return complete edge results and independent matrix products."""
    if len(left) != len(right):
        raise ValueError("Each independent matrix product needs two operands")
    current = torch.cuda.current_stream(source.device)
    side = matrix_stream(source.device, current.cuda_stream)
    side.wait_stream(current)
    count = int(os.environ.get("EQX_XL_PARALLEL_SM_TARGET", "0"))
    from .blas import sm_target

    with torch.cuda.stream(side), sm_target(count) if count else nullcontext():
        products = [selected_mm(a, b) for a, b in zip(left, right)]
    outputs = scheduled_edges(metadata, program, source, target, operands)
    current.wait_stream(side)
    for product in products:
        product.record_stream(current)
    return outputs, products


@parallel_edge_mm.register_fake
def fake_parallel(metadata, program, source, target, operands, left, right):
    return o3_outputs(metadata, program, source, target, operands), [
        a.new_empty((a.shape[0], b.shape[1])) for a, b in zip(left, right)
    ]


@torch.library.custom_op(
    "eqx_oam_xl::parallel_product_mm", mutates_args=(), device_types="cuda"
)
def parallel_product_mm(
    metadata: str,
    program: str,
    operands: list[torch.Tensor],
    left: list[torch.Tensor],
    right: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Overlap the complete node-product program and independent GEMMs."""
    from .packed_product import run_packed
    from .blas import sm_target

    if len(left) != len(right):
        raise ValueError("Each independent matrix product needs two operands")
    current = torch.cuda.current_stream(operands[0].device)
    side = matrix_stream(operands[0].device, current.cuda_stream)
    side.wait_stream(current)
    count = int(os.environ.get("EQX_XL_PARALLEL_SM_TARGET", "0"))
    with torch.cuda.stream(side), sm_target(count) if count else nullcontext():
        products = [selected_mm(a, b) for a, b in zip(left, right)]
    outputs = run_packed(metadata, program, operands)
    current.wait_stream(side)
    for product in products:
        product.record_stream(current)
    return outputs, products


@parallel_product_mm.register_fake
def fake_parallel_product(metadata, program, operands, left, right):
    from .packed_product import packed_outputs

    return packed_outputs(metadata, program, operands), [
        a.new_empty((a.shape[0], b.shape[1])) for a, b in zip(left, right)
    ]


def hint(value):
    """Read a tracing size hint without specializing a symbolic dimension."""
    if isinstance(value, int):
        return value
    return value.node.hint if value.node.hint is not None else 4096


def overlap_parameter_matrices(graph, minimum_k=4096, maximum_matrices=32):
    """Pair independent reduction GEMMs with native edges after autodiff.

    Only GEMMs whose output dimensions are static and reduction dimension is
    sufficiently large are considered. Every operand must already be available.
    Earlier consumers can move after the edge call only when their complete
    dependency closure excludes that edge. Chosen GEMMs must be independent.
    """
    maximum_matrices = int(
        os.environ.get("EQX_XL_PARALLEL_MATRIX_LIMIT", str(maximum_matrices))
    )
    if maximum_matrices < 1:
        raise ValueError("The parallel matrix limit must be positive")
    targets = {"eqx_oam_xl.edge.default", "eqx_oam_xl.pipeline_edge_matrix.default"}
    if os.environ.get("EQX_XL_PARALLEL_PRODUCTS", "0") == "1":
        targets.add("eqx_oam_xl.packed_product.default")
    edge_nodes = [node for node in graph.nodes if str(node.target) in targets]
    count, pairs = 0, []
    for edge in edge_nodes:
        positions = {node: i for i, node in enumerate(graph.nodes)}
        position = positions[edge]
        candidates = []
        for node in graph.nodes:
            if (
                node.target != torch.ops.aten.mm.default
                or node.kwargs
                or len(node.args) != 2
            ):
                continue
            a, b = node.args
            av, bv = a.meta.get("val"), b.meta.get("val")
            if av is None or bv is None or av.ndim != 2 or bv.ndim != 2:
                continue
            if not isinstance(av.shape[0], int) or not isinstance(bv.shape[1], int):
                continue
            if hint(av.shape[1]) < minimum_k or max(av.shape[0], bv.shape[1]) > 16384:
                continue
            if av.dtype not in (torch.float32, torch.float64) or bv.dtype != av.dtype:
                continue
            if any(positions[value] >= position for value in node.all_input_nodes):
                continue
            if not node.users:
                continue
            pending, descendants, dependent = list(node.users), set(), False
            while pending:
                child = pending.pop()
                if child is edge:
                    dependent = True
                    break
                if positions[child] >= position or child in descendants:
                    continue
                descendants.add(child)
                pending.extend(child.users)
            if not dependent:
                candidates.append((node, descendants))
        candidates.sort(
            key=lambda item: (
                hint(item[0].args[0].meta["val"].shape[1]) * item[0].meta["val"].numel()
            ),
            reverse=True,
        )
        selected, moved = [], set()
        for node, descendants in candidates:
            if node in moved or any(old in descendants for old in selected):
                continue
            selected.append(node)
            moved.update(descendants)
            if len(selected) == maximum_matrices:
                break
        candidates = selected
        if not candidates:
            continue
        pipeline = str(edge.target) == "eqx_oam_xl.pipeline_edge_matrix.default"
        product = str(edge.target) == "eqx_oam_xl.packed_product.default"
        if pipeline:
            from .pipeline import pipeline_edge_matrices
        with graph.inserting_before(edge):
            joint = graph.call_function(
                torch.ops.eqx_oam_xl.pipeline_edge_matrices.default
                if pipeline
                else torch.ops.eqx_oam_xl.parallel_product_mm.default
                if product
                else torch.ops.eqx_oam_xl.parallel_edge_mm.default,
                (
                    *edge.args,
                    [n.args[0] for n in candidates],
                    [n.args[1] for n in candidates],
                ),
            )
            joint.meta["val"] = (
                (*edge.meta["val"], [n.meta["val"] for n in candidates])
                if pipeline
                else (edge.meta["val"], [n.meta["val"] for n in candidates])
            )
            joint.meta["eager_input_vals"] = torch.fx.map_arg(
                (joint.args, joint.kwargs), lambda n: n.meta["val"]
            )
            if pipeline:
                values = joint
            else:
                values = graph.call_function(operator.getitem, (joint, 0))
                values.meta["val"] = edge.meta["val"]
            matrices = graph.call_function(
                operator.getitem, (joint, 2 if pipeline else 1)
            )
            matrices.meta["val"] = [n.meta["val"] for n in candidates]
            replacements = []
            for slot, node in enumerate(candidates):
                result = graph.call_function(operator.getitem, (matrices, slot))
                result.meta["val"] = node.meta["val"]
                replacements.append(result)
        anchor = edge
        for node in sorted(moved, key=positions.__getitem__):
            anchor.append(node)
            anchor = node
        edge.replace_all_uses_with(values)
        graph.erase_node(edge)
        for node, result in zip(candidates, replacements):
            node.replace_all_uses_with(result)
            graph.erase_node(node)
        pairs.append(len(candidates))
        count += len(candidates)
    graph.lint()
    return count, pairs
