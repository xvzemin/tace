################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
import operator
from functools import lru_cache
from typing import NamedTuple, Optional

import torch
from e3nn import o3

from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum
from ..time_reversal import contains_time_odd_irreps
from .fused import O3ScatterTensorProduct, uvuTensorProduct
from .paths import satisfy


def _split_tensor_product_inputs(tp: o3.TensorProduct) -> None:
    """Read disjoint feature and weight slices through a shared split."""
    forward = tp._compiled_main_left_right
    if not isinstance(forward, torch.fx.GraphModule):
        return

    slices = {}
    for node in forward.graph.nodes:
        if node.op != "call_function" or node.target is not operator.getitem:
            continue
        tensor, index = node.args
        if not (
            isinstance(tensor, torch.fx.Node)
            and tensor.op == "call_method"
            and tensor.target == "reshape"
            and len(tensor.args) == 3
            and isinstance(tensor.args[2], int)
            and isinstance(index, tuple)
            and len(index) == 2
            and index[0] == slice(None)
            and isinstance(index[1], slice)
            and index[1].step is None
            and isinstance(index[1].start, int)
            and isinstance(index[1].stop, int)
            and index[1].stop > index[1].start
        ):
            continue
        slices.setdefault(tensor, {}).setdefault(
            (index[1].start, index[1].stop), []
        ).append(node)

    for tensor, intervals in slices.items():
        if len(intervals) < 2:
            continue
        sizes = []
        indices = {}
        offset = 0
        for start, stop in sorted(intervals):
            if start < offset:
                break
            if start > offset:
                sizes.append(start - offset)
            indices[start, stop] = len(sizes)
            sizes.append(stop - start)
            offset = stop
        else:
            if offset < tensor.args[2]:
                sizes.append(tensor.args[2] - offset)
            with forward.graph.inserting_after(tensor):
                split = forward.graph.call_function(
                    torch.split, (tensor, sizes), {"dim": 1}
                )
            for interval, nodes in intervals.items():
                for node in nodes:
                    node.args = (split, indices[interval])

    forward.graph.eliminate_dead_code()
    forward.graph.lint()
    forward.recompile()


@lru_cache(maxsize=None)
def wigner_6j(
    l1: int,
    l2: int,
    l1l2: int,
    l3: int,
    L: int,
    l23: int,
) -> float:
    r"""Wigner-6j coefficient for exchanging the edge and extra-node inputs.

    Parameters
    ----------
    l1 : int
        Angular degree of the edge input.
    l2 : int
        Angular degree of the node input.
    l1l2 : int
        Angular degree of the node-edge intermediate.
    l3 : int
        Angular degree of the extra node input.
    L : int
        Angular degree of the output.
    l23 : int
        Angular degree of the node-extra intermediate.

    Returns
    -------
    float
        Recoupling coefficient, including the phase and dimension factors.

    Notes
    -----
    The coefficient is

    .. math::

        (-1)^{l_1+l_3+l_{12}+l_{23}}
        \sqrt{(2l_{12}+1)(2l_{23}+1)}
        \begin{Bmatrix}
            l_1 & l_2 & l_{12} \\
            l_3 & L & l_{23}
        \end{Bmatrix}.

    It converts ``(node x edge) x extra`` to ``(node x extra) x edge``
    with component-normalized Clebsch-Gordan products. Path normalization
    is applied separately by :class:`O3Wigner6jScatterTensorProduct`.
    """
    from sympy.physics.wigner import wigner_6j as _wigner_6j

    return (
        (-1) ** (l1 + l3 + l1l2 + l23)
        * math.sqrt((2 * l1l2 + 1) * (2 * l23 + 1))
        * float(_wigner_6j(l1, l2, l1l2, l3, L, l23))
    )


class _CouplingPath(NamedTuple):
    i_in1: int
    i_in2: int
    i_in3: int
    ir12: o3.Irrep
    ir_out: o3.Irrep
    mul: int
    weight_offset: int


class O3Wigner6jScatterTensorProduct(torch.nn.Module):
    r"""Three-input tensor product with Wigner-6j recoupling and node aggregation.

    Parameters
    ----------
    irreps_node_feats : `e3nn.o3.Irreps`
        Irreps of the node features.
    irreps_edge_attrs : `e3nn.o3.Irreps`
        Irreps of the edge attributes. Each multiplicity must be one.
    irreps_out : `e3nn.o3.Irreps`
        Requested output irrep types. Each allowed coupling path produces a
        separate output with the multiplicity of its node input; the supplied
        output multiplicities are not used.
    extra_irreps_node_attrs : `e3nn.o3.Irreps`
        Irreps of the extra node attributes. Each multiplicity must be one.
    weight_level : {'edge', 'node'}
        Whether ``extra_weights`` are supplied per edge or per source node.
        ``edge_weights`` are always supplied per edge.
    l1l2 : {'<', '<=', '>', '>=', '==', '!='} or None
        Restriction on the angular degrees of the node and edge inputs.
        If ``None``, all allowed node-edge pairs are included.
    register_reference : bool, default False
        Retain the unrecoupled tensor products for :meth:`forward_reference`.

    Attributes
    ----------
    irreps_out : `e3nn.o3.Irreps`
        Output irreps in coupling-path order.
    edge_weight_numel : int
        Number of edge weights per edge.
    extra_weight_numel : int
        Number of extra weights per edge or node.

    Notes
    -----
    The reference tree is ``(node_feats x edge_attrs) x extra_node_attrs``;
    the executed tree is ``(node_feats x extra_node_attrs) x edge_attrs``.
    Fixed Wigner-6j coefficients and path-normalization factors make the two
    trees equivalent. Each distinct node-node coupling is evaluated once.
    Independent reference paths retain separate weights and output channels.
    Both weights are applied in the final node-edge product, whose results
    are summed at the target nodes.
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_out: o3.Irreps,
        extra_irreps_node_attrs: o3.Irreps,
        *,
        weight_level: str,
        l1l2: Optional[str] = None,
        register_reference: bool = False,
    ) -> None:
        super().__init__()

        if weight_level not in {"edge", "node"}:
            raise ValueError(
                f"weight_level must be either 'edge' or 'node', got {weight_level!r}"
            )

        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        irreps_out = o3.Irreps(irreps_out)
        self.extra_irreps_node_attrs = o3.Irreps(extra_irreps_node_attrs)
        if any(mul != 1 for mul, _ in self.extra_irreps_node_attrs):
            raise ValueError("extra_irreps_node_attrs must have multiplicity one")
        if contains_time_odd_irreps(
            self.irreps_node_feats,
            self.irreps_edge_attrs,
            irreps_out,
            self.extra_irreps_node_attrs,
        ):
            for kernel in ("oeq", "cue"):
                if acceleration_enabled(kernel):
                    raise ValueError(
                        f"{kernel.upper()} does not support time-reversal "
                        "Wigner-6j scatter tensor products. Disable the "
                        "accelerated scatter kernel."
                    )
        self.weight_level = weight_level
        self.register_reference = register_reference

        paths: list[_CouplingPath] = []
        reference_irrep_list = []
        irrep_list_out = []
        reference_node_edge_instructions = []
        reference_edge_edge_instructions = []
        weight_offset = 0

        for _, ir_out in irreps_out:
            for i1, (mul, ir1) in enumerate(self.irreps_node_feats):
                for i2, (_, ir2) in enumerate(self.irreps_edge_attrs):
                    if not satisfy(ir1.l, ir2.l, l1l2):
                        continue
                    for ir12 in ir1 * ir2:
                        for i3, (_, ir3) in enumerate(self.extra_irreps_node_attrs):
                            if ir_out not in ir12 * ir3:
                                continue

                            i_out = len(paths)
                            paths.append(
                                _CouplingPath(
                                    i1, i2, i3, ir12, ir_out, mul, weight_offset
                                )
                            )
                            reference_irrep_list.append((mul, ir12))
                            irrep_list_out.append((mul, ir_out))
                            reference_node_edge_instructions.append(
                                (i1, i2, i_out, "uvu", True, 1.0)
                            )
                            reference_edge_edge_instructions.append(
                                (i_out, i3, i_out, "uvu", True, 1.0)
                            )
                            weight_offset += mul

        if not paths:
            raise ValueError("No Wigner-6j coupling paths were generated")

        self.irreps_out = o3.Irreps(irrep_list_out)

        reference_node_edge_tp = o3.TensorProduct(
            self.irreps_node_feats,
            self.irreps_edge_attrs,
            o3.Irreps(reference_irrep_list),
            reference_node_edge_instructions,
            internal_weights=False,
            shared_weights=False,
        )
        reference_edge_edge_tp = o3.TensorProduct(
            o3.Irreps(reference_irrep_list),
            self.extra_irreps_node_attrs,
            self.irreps_out,
            reference_edge_edge_instructions,
            internal_weights=False,
            shared_weights=False,
        )

        recoupled_irrep_list = []
        intermediate_indices = {}
        recoupled_node_node_instructions = []
        recoupled_node_edge_instructions = []
        source_weight_indices = []
        recoupling_path_indices = []
        component_recoupling_coefficients = []

        for i_out, path in enumerate(paths):
            ir1 = self.irreps_node_feats[path.i_in1].ir
            ir2 = self.irreps_edge_attrs[path.i_in2].ir
            ir3 = self.extra_irreps_node_attrs[path.i_in3].ir
            for ir13 in ir1 * ir3:
                if path.ir_out not in ir13 * ir2:
                    continue

                coefficient = wigner_6j(
                    ir2.l,
                    ir1.l,
                    path.ir12.l,
                    ir3.l,
                    path.ir_out.l,
                    ir13.l,
                )
                if abs(coefficient) < 1.0e-14:
                    continue

                key = (path.i_in1, path.i_in3, ir13)
                if key not in intermediate_indices:
                    intermediate_indices[key] = len(recoupled_irrep_list)
                    recoupled_irrep_list.append((path.mul, ir13))
                    recoupled_node_node_instructions.append(
                        (
                            path.i_in1,
                            path.i_in3,
                            intermediate_indices[key],
                            "uvu",
                            True,
                            1.0,
                        )
                    )
                intermediate_index = intermediate_indices[key]
                recoupled_node_edge_instructions.append(
                    (
                        intermediate_index,
                        path.i_in2,
                        i_out,
                        "uvu",
                        True,
                        1.0,
                    )
                )
                source_weight_indices.extend(
                    range(
                        path.weight_offset,
                        path.weight_offset + path.mul,
                    )
                )
                recoupling_path_indices.append(i_out)
                component_recoupling_coefficients.append(coefficient)

        # Produce equal irreps contiguously, without a runtime permutation or
        # summing independent channels. uvu instructions retain their slices.
        irreps_mid, permutation, _ = o3.Irreps(recoupled_irrep_list).sort()
        recoupled_node_node_instructions = sorted(
            (
                (i, j, permutation[k], *rest)
                for i, j, k, *rest in recoupled_node_node_instructions
            ),
            key=lambda ins: ins[2],
        )
        recoupled_node_edge_instructions = [
            (permutation[i], j, k, *rest)
            for i, j, k, *rest in recoupled_node_edge_instructions
        ]

        self.recoupled_node_node_tp = uvuTensorProduct(
            self.irreps_node_feats,
            self.extra_irreps_node_attrs,
            irreps_mid,
            instructions=recoupled_node_node_instructions,
            shared_weights=True,
        )
        self.recoupled_node_edge_tp = O3ScatterTensorProduct(
            irreps_mid,
            self.irreps_edge_attrs,
            self.irreps_out,
            instructions=recoupled_node_edge_instructions,
        )
        for tp in (self.recoupled_node_node_tp, self.recoupled_node_edge_tp):
            if not hasattr(tp, "fused_tp"):
                _split_tensor_product_inputs(tp.tp)

        recoupling_coefficients = []
        for ins, i_out, coefficient in zip(
            self.recoupled_node_edge_tp.tp.instructions,
            recoupling_path_indices,
            component_recoupling_coefficients,
        ):
            path = paths[i_out]
            ir13 = irreps_mid[ins.i_in1].ir
            reference_scale = (
                reference_node_edge_tp.instructions[i_out].path_weight
                * reference_edge_edge_tp.instructions[i_out].path_weight
            )
            recoupled_scale = (
                self.recoupled_node_node_tp.instructions[ins.i_in1].path_weight
                * ins.path_weight
            )
            reference_component_scale = math.sqrt(path.ir12.dim * path.ir_out.dim)
            recoupled_component_scale = math.sqrt(ir13.dim * path.ir_out.dim)
            reference_element_scale = reference_scale / reference_component_scale
            recoupled_element_scale = recoupled_scale / recoupled_component_scale
            normalized_coefficient = (
                coefficient * reference_element_scale / recoupled_element_scale
            )
            recoupling_coefficients.extend([normalized_coefficient] * path.mul)

        self.edge_weight_numel = reference_node_edge_tp.weight_numel
        self.extra_weight_numel = reference_edge_edge_tp.weight_numel
        if self.edge_weight_numel != weight_offset:
            raise RuntimeError("Unexpected e3nn edge weight layout")
        if self.extra_weight_numel != weight_offset:
            raise RuntimeError("Unexpected e3nn extra-node-attribute weight layout")

        if self.register_reference:
            self.reference_node_edge_tp = reference_node_edge_tp
            self.reference_edge_edge_tp = reference_edge_edge_tp

        self.register_buffer(
            "node_node_weights",
            torch.ones(self.recoupled_node_node_tp.weight_numel),
            persistent=False,
        )
        self.register_buffer(
            "source_weight_indices",
            torch.tensor(source_weight_indices, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "recoupling_coefficients",
            torch.tensor(recoupling_coefficients, dtype=torch.float64),
            persistent=False,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # This derived mask changes when repeated intermediates are shared;
        # learned weights and all other tensor-product buffers are unchanged.
        key = f"{prefix}recoupled_node_node_tp.tp.output_mask"
        if key in state_dict:
            state_dict[key] = self.recoupled_node_node_tp.tp.output_mask
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        extra_node_attrs: torch.Tensor,
        edge_weights: torch.Tensor,
        extra_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the recoupled tensor product and sum at target nodes.

        Parameters
        ----------
        node_feats : torch.Tensor
            Node features of shape ``(num_nodes, irreps_node_feats.dim)``.
        edge_attrs : torch.Tensor
            Edge attributes of shape ``(num_edges, irreps_edge_attrs.dim)``.
        extra_node_attrs : torch.Tensor
            Extra node attributes of shape
            ``(num_nodes, extra_irreps_node_attrs.dim)``.
        edge_weights : torch.Tensor
            External weights of shape ``(num_edges, edge_weight_numel)``.
        extra_weights : torch.Tensor
            External weights of shape ``(num_edges, extra_weight_numel)``
            when ``weight_level='edge'``, or ``(num_nodes, extra_weight_numel)``
            when ``weight_level='node'``.
        edge_index : torch.Tensor
            Integer indices of shape ``(2, num_edges)``. Row zero contains
            source nodes and row one contains target nodes.

        Returns
        -------
        torch.Tensor
            Node features of shape ``(num_nodes, irreps_out.dim)``.
        """
        coefficients = self.recoupling_coefficients.to(dtype=edge_weights.dtype)
        if self.weight_level == "edge":
            if extra_weights.size(0) != edge_index.size(1):
                raise ValueError("edge weights must have one row per graph edge")
        else:
            if extra_weights.size(0) != node_feats.size(0):
                raise ValueError("node weights must have one row per graph node")
            extra_weights = extra_weights.index_select(0, edge_index[0])
        # Scalar path weights commute with the CG contraction. Combine them
        # before expanding the reference paths into recoupling paths.
        edge_weights = (edge_weights * extra_weights).index_select(
            -1, self.source_weight_indices
        )
        node_node_intermediate = self.recoupled_node_node_tp(
            node_feats,
            extra_node_attrs,
            self.node_node_weights.to(dtype=node_feats.dtype),
        )
        return self.recoupled_node_edge_tp(
            node_node_intermediate,
            edge_attrs,
            edge_weights * coefficients,
            edge_index,
        )

    def forward_reference(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        extra_node_attrs: torch.Tensor,
        edge_weights: torch.Tensor,
        extra_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the unrecoupled tensor product and sum at target nodes.

        Inputs and output have the same meaning and shapes as in :meth:`forward`.

        Returns
        -------
        torch.Tensor
            Node features of shape ``(num_nodes, irreps_out.dim)``.

        Raises
        ------
        RuntimeError
            If the module was constructed with ``register_reference=False``.

        See Also
        --------
        forward : Evaluate the same operation using Wigner-6j recoupling.
        """
        if not self.register_reference:
            raise RuntimeError(
                "forward_reference requires register_reference=True at construction"
            )

        source = edge_index[0]
        node_edge_intermediate = self.reference_node_edge_tp(
            node_feats[source], edge_attrs, edge_weights
        )
        if self.weight_level == "edge":
            if extra_weights.size(0) != edge_index.size(1):
                raise ValueError("edge weights must have one row per graph edge")
        else:
            if extra_weights.size(0) != node_feats.size(0):
                raise ValueError("node weights must have one row per graph node")
            extra_weights = extra_weights[source]
        message = self.reference_edge_edge_tp(
            node_edge_intermediate,
            extra_node_attrs[source],
            extra_weights,
        )
        return scatter_sum(
            message,
            edge_index[1],
            dim=0,
            dim_size=node_feats.size(0),
        )
