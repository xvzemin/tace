"""Overlap a future ready matrix product with the current edge program.

Scheduling acts on the explicit joint derivative graph. Only pure metadata
views and bounded parameter arithmetic may move ahead of an edge, and every
CUDA fork is joined inside a single dispatcher operation.
"""
from contextlib import nullcontext
import operator
import os

import torch

from equivariantx.utils.outputs import o3_outputs
from .edge import scheduled_edges
from .parallel import hint, matrix_stream


@torch.library.custom_op('eqx_oam_xl::pipeline_edge_matrix', mutates_args=(), device_types='cuda')
def pipeline_edge_matrix(metadata: str, program: str, source: torch.Tensor,
                         target: torch.Tensor, operands: list[torch.Tensor],
                         left: torch.Tensor, right: torch.Tensor,
                         bias: torch.Tensor | None) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Execute independent CG and GEMM branches, joining before returning."""
    current = torch.cuda.current_stream(source.device)
    side = matrix_stream(source.device, current.cuda_stream)
    side.wait_stream(current)
    count = int(os.environ.get('EQX_XL_PARALLEL_SM_TARGET', '0'))
    from .blas import sm_target
    with torch.cuda.stream(side), (sm_target(count) if count else nullcontext()):
        product = torch.mm(left, right) if bias is None else torch.addmm(bias, left, right)
    outputs = scheduled_edges(metadata, program, source, target, operands)
    current.wait_stream(side)
    product.record_stream(current)
    return outputs, product


@pipeline_edge_matrix.register_fake
def fake_pipeline(metadata, program, source, target, operands, left, right, bias):
    return o3_outputs(metadata, program, source, target, operands), left.new_empty((left.shape[0], right.shape[1]))


@torch.library.custom_op('eqx_oam_xl::pipeline_edge_matrices', mutates_args=(), device_types='cuda')
def pipeline_edge_matrices(metadata: str, program: str, source: torch.Tensor,
                           target: torch.Tensor, operands: list[torch.Tensor],
                           left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor | None,
                           extra_left: list[torch.Tensor], extra_right: list[torch.Tensor]
                           ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """Retain parameter-GEMM overlap while adding a future activation GEMM."""
    if len(extra_left) != len(extra_right):
        raise ValueError('Independent GEMMs require paired operands')
    current = torch.cuda.current_stream(source.device)
    side = matrix_stream(source.device, current.cuda_stream)
    side.wait_stream(current)
    count = int(os.environ.get('EQX_XL_PARALLEL_SM_TARGET', '0'))
    from .blas import sm_target
    with torch.cuda.stream(side), (sm_target(count) if count else nullcontext()):
        product = torch.mm(left, right) if bias is None else torch.addmm(bias, left, right)
        extras = [torch.mm(a,b) for a,b in zip(extra_left,extra_right)]
    outputs = scheduled_edges(metadata, program, source, target, operands)
    current.wait_stream(side)
    for value in (product, *extras):
        value.record_stream(current)
    return outputs, product, extras


@pipeline_edge_matrices.register_fake
def fake_pipeline_matrices(metadata, program, source, target, operands, left, right, bias, extra_left, extra_right):
    return (o3_outputs(metadata, program, source, target, operands),
            left.new_empty((left.shape[0], right.shape[1])),
            [a.new_empty((a.shape[0],b.shape[1])) for a,b in zip(extra_left,extra_right)])


def overlap_future_matrices(graph, minimum_flops=200_000_000):
    """Move one independent future GEMM beside each available edge program.

    Parameters
    ----------
    graph : torch.fx.Graph
        Functional, already differentiated graph with fake tensor metadata.
    minimum_flops : int
        Estimated arithmetic threshold used only to select an execution order.

    Returns
    -------
    list of tuple[str, str]
        Original edge and GEMM names, for diagnostics.
    """
    view_ops = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default,
                torch.ops.aten.reshape.default, torch.ops.aten.t.default,
                torch.ops.aten.transpose.int, torch.ops.aten.permute.default,
                torch.ops.aten.select.int, torch.ops.aten.slice.Tensor,
                torch.ops.aten.detach.default}
    arithmetic = {torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor,
                  torch.ops.aten.div.Tensor, torch.ops.aten.pow.Tensor_Scalar}
    edges = [n for n in graph.nodes if str(n.target) == 'eqx_oam_xl.edge.default']
    selected = []
    for edge in edges:
        positions = {n:i for i,n in enumerate(graph.nodes)}
        at = positions[edge]
        candidates = []
        for node in graph.nodes:
            if (positions[node] <= at or not node.users or node.kwargs
                    or node.target not in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)):
                continue
            bias, left, right = ((None, *node.args) if node.target == torch.ops.aten.mm.default else node.args)
            a, b = left.meta.get('val'), right.meta.get('val')
            if (a is None or b is None or a.ndim != 2 or b.ndim != 2
                    or a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype):
                continue
            cost = 2*hint(a.shape[0])*hint(a.shape[1])*hint(b.shape[1])
            if cost < minimum_flops:
                continue
            moving = set()
            def available(parent):
                if positions[parent] < at:
                    return True
                if parent in moving:
                    return True
                if parent.target not in view_ops | arithmetic:
                    return False
                value = parent.meta.get('val')
                if parent.target in arithmetic:
                    if (not isinstance(value, torch.Tensor)
                            or any(not isinstance(d, int) for d in value.shape)
                            or value.numel() > 4_000_000):
                        return False
                if not all(available(p) for p in parent.all_input_nodes):
                    return False
                moving.add(parent)
                return True
            if all(available(parent) for parent in node.all_input_nodes):
                candidates.append((node, moving, bias, left, right))
        if not candidates:
            continue
        node, moving, bias, left, right = candidates[0]
        for parent in sorted(moving, key=positions.__getitem__):
            edge.prepend(parent)
        with graph.inserting_before(edge):
            joint = graph.call_function(torch.ops.eqx_oam_xl.pipeline_edge_matrix.default,
                                        (*edge.args, left, right, bias))
            joint.meta['val'] = (edge.meta['val'], node.meta['val'])
            joint.meta['eager_input_vals'] = torch.fx.map_arg((joint.args, joint.kwargs), lambda n:n.meta['val'])
            edges_out = graph.call_function(operator.getitem, (joint, 0))
            edges_out.meta['val'] = edge.meta['val']
            matrix = graph.call_function(operator.getitem, (joint, 1))
            matrix.meta['val'] = node.meta['val']
        selected.append((edge.name, node.name))
        edge.replace_all_uses_with(edges_out)
        node.replace_all_uses_with(matrix)
        graph.erase_node(edge)
        graph.erase_node(node)
    graph.lint()
    return selected
