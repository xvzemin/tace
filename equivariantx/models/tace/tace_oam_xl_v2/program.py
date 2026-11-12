"""Whole-training scheduling of explicit mixed parameter derivatives.

A force-training graph often computes node and weight adjoints during the tangent
sweep, computes its second contribution during the final reverse sweep, then
adds the two arrays. When the early result has no other consumer, it can be
evaluated in the later native program. This preserves all terms while avoiding
one materialized output and its subsequent full-array addition.
"""

import hashlib
import logging
import operator
import os
from pathlib import Path

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass

from equivariantx.utils.metadata import parse_metadata

OPERANDS = {
    "eqx.o3_contraction.default": 4,
    "eqx_oam_xl.edge.default": 4,
    "eqx_oam_xl.compressed_edge.default": 4,
    "eqx_oam_xl.linear.default": 2,
    "eqx_oam_xl.packed_product.default": 2,
}


def refresh_inputs(node):
    """Keep Inductor's recorded Tensor[] layout constraints in sync with FX.

    Inductor uses ``eager_input_vals`` to constrain custom-op argument layouts.
    Keeping the old shorter list silently drops newly appended operands during
    lowering, even though executing the rewritten FX graph itself is correct.
    """
    if "eager_input_vals" in node.meta:
        node.meta["eager_input_vals"] = torch.fx.map_arg(
            (node.args, node.kwargs), lambda value: value.meta["val"]
        )


def selected_terms(node, slot):
    """Separate one adjoint output from the native program."""
    program = parse_metadata(node.args[1])
    selected, remaining, owners = [], [], set()
    if str(node.target) == "eqx_oam_xl.packed_product.default":
        for mapping, role, index, required in program:
            if index == slot:
                selected.append((mapping, role, index, required))
                owners.add(role)
            else:
                remaining.append((mapping, role, index, required))
    elif str(node.target) == "eqx_oam_xl.linear.default":
        for path, mapping, role, index in program:
            if index == slot:
                selected.append((path, mapping, role, index))
                owners.add(role)
            else:
                remaining.append((path, mapping, role, index))
    else:
        for mapping, weighted, pairs in program:
            take = tuple(p for p in pairs if p[1] == slot)
            keep = tuple(p for p in pairs if p[1] != slot)
            if take:
                selected.append((mapping, weighted, take))
                owners.update(role for role, _ in take)
            if keep:
                remaining.append((mapping, weighted, keep))
    return selected, remaining, owners


def remap_terms(terms, linear, pointers, slots, packed=False):
    if packed:
        return tuple((tuple(pointers[i] for i in mapping), role, slots[slot], required)
                     for mapping, role, slot, required in terms)
    if linear:
        return tuple(
            (path, tuple(pointers[i] for i in mapping), role, slots[slot])
            for path, mapping, role, slot in terms
        )
    return tuple(
        (
            tuple(pointers[i] for i in mapping),
            weighted,
            tuple((role, slots[slot]) for role, slot in pairs),
        )
        for mapping, weighted, pairs in terms
    )


def fuse_gradient_sums(graph):
    """Delay exclusively consumed node and weight adjoints into their final sum.

    The pass only accepts unit-coefficient sums, identical operator contracts,
    matching graph indices, and single-use intermediate outputs. Other consumers
    keep their existing schedule. In a mixed edge sweep this separates the
    target-node tangent from the source-node adjoint: each sweep then reduces
    on its own graph direction instead of issuing cross-direction atomics.
    """
    positions = {node: index for index, node in enumerate(graph.nodes)}
    count = 0
    for addition in list(graph.nodes):
        if (
            addition.target != torch.ops.aten.add.Tensor
            or len(addition.args) != 2
            or addition.kwargs.get("alpha", 1) != 1
        ):
            continue
        a, b = addition.args
        if not all(
            isinstance(x, torch.fx.Node)
            and x.target == operator.getitem
            and len(x.args) == 2
            and isinstance(x.args[1], int)
            and len(x.users) == 1
            for x in (a, b)
        ):
            continue
        first, second = a.args[0], b.args[0]
        if (
            first is second
            or first.target != second.target
            or str(first.target) not in OPERANDS
            or first.kwargs != second.kwargs
            or first.args[0] != second.args[0]
        ):
            continue
        if positions[first] > positions[second]:
            first, second, a, b = second, first, b, a
        offset = OPERANDS[str(first.target)]
        if offset == 4 and first.args[2:4] != second.args[2:4]:
            continue
        if str(first.target) == "eqx_oam_xl.compressed_edge.default" and first.args[5] != second.args[5]:
            continue
        if any(
            user.target != operator.getitem or not isinstance(user.args[1], int)
            for user in first.users
        ):
            continue
        first_slot, second_slot = a.args[1], b.args[1]
        if sum(user.args[1] == first_slot for user in first.users) != 1:
            continue
        selected, remaining, owners = selected_terms(first, first_slot)
        other_selected, _, other_owners = selected_terms(second, second_slot)
        packed = str(first.target) == "eqx_oam_xl.packed_product.default"
        linear = str(first.target) == "eqx_oam_xl.linear.default"
        allowed = (bool(owners) and (owners <= {0, 1} or owners == {2})) if packed else (len(owners) == 1 and owners <= {0, 1})
        if not allowed or other_owners != owners:
            continue
        if linear and {term[0] for term in selected} != {
            term[0] for term in other_selected
        }:
            continue
        av, bv = a.meta.get("val"), b.meta.get("val")
        if (
            av is None
            or bv is None
            or av.dtype != bv.dtype
            or av.device != bv.device
            or tuple(map(str, av.shape)) != tuple(map(str, bv.shape))
        ):
            continue
        values = list(second.args[offset])
        pointers = {}
        for i, value in enumerate(first.args[offset]):
            if value not in values:
                values.append(value)
            pointers[i] = values.index(value)
        moved = remap_terms(selected, linear, pointers, {first_slot: second_slot}, packed)
        arguments = list(second.args)
        arguments[1] = repr(parse_metadata(arguments[1]) + moved)
        arguments[offset] = values
        second.args = tuple(arguments)
        refresh_inputs(second)
        addition.replace_all_uses_with(b)
        graph.erase_node(addition)
        graph.erase_node(a)
        if not remaining:
            graph.erase_node(first)
        else:
            slots = sorted(
                {term[2] for term in remaining} if packed else {term[3] for term in remaining}
                if linear
                else {slot for _, _, pairs in remaining for _, slot in pairs}
            )
            slots = {old: new for new, old in enumerate(slots)}
            arguments = list(first.args)
            arguments[1] = repr(
                remap_terms(
                    remaining,
                    linear,
                    dict(enumerate(range(len(arguments[offset])))),
                    slots,
                    packed,
                )
            )
            first.args = tuple(arguments)
            refresh_inputs(first)
            for user in list(first.users):
                user.args = (first, slots[user.args[1]])
            for key in ("val", "example_value"):
                if key in first.meta:
                    first.meta[key] = [first.meta[key][old] for old in slots]
            for user in first.users:
                refresh_inputs(user)
        count += 1
    graph.lint()
    return count


class MixedGradientPass(CustomGraphPass):
    """Inductor hook with source-versioned cache identity."""

    def __call__(self, graph):
        from .precision import validate_environment, validate_graph
        from .extensions import apply_graph_extensions

        validate_environment()
        validate_graph(graph)
        apply_graph_extensions(graph)
        validate_graph(graph)
        count = fuse_gradient_sums(graph)
        logging.info("TACE-OAM-XL-v2 fused %d mixed-gradient sums", count)
        if os.environ.get('EQX_XL_ORDERED_REDUCTIONS', '0') == '1':
            from .ordered_reductions import order_reductions
            logging.info('TACE-OAM-XL-v2 stable node reductions: %s', order_reductions(graph))
        if os.environ.get('EQX_XL_PACKED_FRAGMENTS', '0') == '1':
            from .packed_fragments import fragment_packed_graph
            logging.info('TACE-OAM-XL-v2 direct cotangent blocks: %s', fragment_packed_graph(graph))
        if os.environ.get('EQX_XL_JOIN_MATRICES', '0') == '1':
            from .joined_matrix import join_matrix_graph
            logging.info('TACE-OAM-XL-v2 joined live matrices: %s', join_matrix_graph(graph))
        matrices = os.environ.get("EQX_XL_PARALLEL_MATRICES", "0") == "1"
        pipeline = os.environ.get("EQX_XL_PIPELINE_MATRICES", "0") == "1"
        if matrices or pipeline:
            from torch._inductor import config
            if config.shape_padding:
                raise RuntimeError(
                    "TACE-OAM-XL-v2 parallel scheduling requires shape_padding=False: "
                    "the installed compiler reuses a padded BMM buffer while a "
                    "gradient view is still live. Use compiler-no-shape-padding.json."
                )
            if pipeline:
                from .pipeline import overlap_future_matrices
                pairs = overlap_future_matrices(graph)
                logging.info("TACE-OAM-XL-v2 pipelined %d future GEMMs: %s", len(pairs), pairs)
            if matrices:
                from .parallel import overlap_parameter_matrices
                count, pairs = overlap_parameter_matrices(graph)
                logging.info("TACE-OAM-XL-v2 overlapped %d parameter GEMMs in %s groups", count, pairs)
        if os.environ.get('EQX_XL_LT_RADIAL','0') == '1':
            from .lt_matrix import select_radial_matrices
            logging.info('TACE-OAM-XL-v2 calibrated radial GEMMs: %s',select_radial_matrices(graph))

    def uuid(self):
        from .extensions import extension_identity
        sources = [Path(__file__).read_bytes(), Path(__file__).with_name("parallel.py").read_bytes(),
                   Path(__file__).with_name("pipeline.py").read_bytes()]
        sources.append(Path(__file__).with_name("node_sweep.py").read_bytes())
        sources.append(Path(__file__).with_name("packed_fragments.py").read_bytes())
        sources.append(Path(__file__).with_name("ordered_reductions.py").read_bytes())
        sources.append(os.environ.get('EQX_XL_ORDERED_REDUCTIONS', '0').encode())
        sources.append(os.environ.get('EQX_XL_PACKED_FRAGMENTS', '0').encode())
        sources.append(os.environ.get('EQX_XL_NODE_SWEEP', '0').encode())
        sources.append(os.environ.get('EQX_XL_NODE_TANGENT', '0').encode())
        sources.append(Path(__file__).with_name("source_vjp.py").read_bytes())
        sources.append(os.environ.get('EQX_XL_SOURCE_VJP', '0').encode())
        sources.append(os.environ.get('EQX_XL_SOURCE_VJP_REGISTER_LIMIT', '255').encode())
        sources.append(os.environ.get('EQX_XL_MIXED_REUSE_ANGULAR', '0').encode())
        sources.append(Path(__file__).with_name('row_schedule.py').read_bytes())
        sources.append(Path(__file__).with_name('lt_matrix.py').read_bytes())
        sources.append(os.environ.get('EQX_XL_LT_RADIAL','0').encode())
        sources.append(os.environ.get('EQX_XL_LT_SM_TARGET','0').encode())
        sources.append(os.environ.get('EQX_XL_LT_ALGORITHM','-1').encode())
        sources.append(os.environ.get('EQX_XL_SOURCE_ROW_LIMIT', '0').encode())
        sources.append(os.environ.get('EQX_XL_MIXED_ROW_LIMIT', '0').encode())
        sources.append(os.environ.get('EQX_XL_MIXED_OWNER', 'off').encode())
        sources.append(Path(__file__).with_name("joined_matrix.py").read_bytes())
        sources.append(os.environ.get('EQX_XL_JOIN_MATRICES', '0').encode())
        sources.append(os.environ.get("EQX_XL_PARALLEL_MATRICES", "0").encode())
        sources.append(os.environ.get("EQX_XL_PARALLEL_PRODUCTS", "0").encode())
        sources.append(os.environ.get("EQX_XL_EDGE_ANGULAR_BLOCK", "0").encode())
        sources.append(os.environ.get("EQX_XL_LAUNCH_CONTRACT", "0").encode())
        sources.append(os.environ.get("EQX_XL_PARALLEL_MATRIX_LIMIT", "32").encode())
        sources.append(os.environ.get("EQX_XL_CG_THREADS", "0").encode())
        sources.append(os.environ.get("EQX_XL_PIPELINE_MATRICES", "0").encode())
        sources.extend(Path(__file__).with_name(name).read_bytes() for name in (
            "precision.py", "extensions.py", "cg_codegen.py", "cg_convolution.py", "graph.py"
        ))
        sources.append(extension_identity().encode())
        return hashlib.sha256(b"\0".join(sources)).hexdigest()
