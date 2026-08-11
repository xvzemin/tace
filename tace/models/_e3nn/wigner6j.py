################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Wigner-6j recoupling for O(3) interactions."""

import math
from dataclasses import dataclass
from typing import Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from .fused import O3ScatterTensorProduct
from .paths import satisfy


def sympy_wigner_6j(
    l1: int,
    l2: int,
    l1l2: int,
    l3: int,
    L: int,
    l23: int,
) -> float:

    from sympy import S
    from sympy.physics import wigner

    return float(
        wigner.wigner_6j(
            S(l1),
            S(l2),
            S(l1l2),
            S(l3),
            S(L),
            S(l23),
        )
    )


def wigner_6j(
    l1: int,
    l2: int,
    l1l2: int,
    l3: int,
    L: int,
    l23: int,
) -> float:
    r"""The generic angular-momentum labels map to:

    - ``l1``: edge spherical harmonic
    - ``l2``: node feature
    - ``l1l2``: node-edge intermediate
    - ``l3``: extra node attribute
    - ``L``: output
    - ``l23``: node-extra-attribute intermediate
    """

    return (
        (-1) ** (l1 + l3 + l1l2 + l23)
        * math.sqrt((2 * l1l2 + 1) * (2 * l23 + 1))
        * sympy_wigner_6j(
            l1,
            l2,
            l1l2,
            l3,
            L,
            l23,
        )
    )


@dataclass(frozen=True)
class _CouplingPath:
    node_index: int
    edge_index: int
    extra_index: int
    node_edge_irrep: o3.Irrep
    out_irrep: o3.Irrep
    multiplicity: int
    weight_offset: int


class O3Wigner6jScatterTensorProduct(torch.nn.Module):
    r"""
    The reference tree is ``(node_feats x edge_attrs) x extra_node_attrs``. The
    executed tree is ``(node_feats x extra_node_attrs) x edge_attrs``. Every
    complete reference path remains separate, and all allowed first intermediate
    irreps are summed with fixed Wigner-6j coefficients. Therefore the two trees
    are algebraically identical.
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_out: o3.Irreps,
        extra_irreps_node_attrs: o3.Irreps,
        *,
        weight_level: str,
        l1l2: Union[str, None] = None,
        register_reference: bool = False,
    ) -> None:
        super().__init__()

        if weight_level not in {"edge", "node"}:
            raise ValueError(
                f"weight_level must be either 'edge' or 'node', got {weight_level!r}"
            )

        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        requested_irreps_out = o3.Irreps(irreps_out)
        self.extra_irreps_node_attrs = o3.Irreps(extra_irreps_node_attrs)
        if any(multiplicity != 1 for multiplicity, _ in self.extra_irreps_node_attrs):
            raise ValueError("extra_irreps_node_attrs must have multiplicity one")
        self.weight_level = weight_level
        self.register_reference = register_reference

        paths: list[_CouplingPath] = []
        reference_intermediate = []
        expanded_output = []
        reference_node_edge_instructions = []
        reference_edge_edge_instructions = []
        weight_offset = 0

        for _, (_, out_irrep) in enumerate(requested_irreps_out):
            for node_index, (multiplicity, node_irrep) in enumerate(
                self.irreps_node_feats
            ):
                for edge_index, (_, edge_irrep) in enumerate(self.irreps_edge_attrs):
                    if not satisfy(node_irrep.l, edge_irrep.l, l1l2):
                        continue
                    for node_edge_irrep in node_irrep * edge_irrep:
                        for extra_index, (_, extra_irrep) in enumerate(
                            self.extra_irreps_node_attrs
                        ):
                            if out_irrep not in node_edge_irrep * extra_irrep:
                                continue

                            path_index = len(paths)
                            paths.append(
                                _CouplingPath(
                                    node_index=node_index,
                                    edge_index=edge_index,
                                    extra_index=extra_index,
                                    node_edge_irrep=node_edge_irrep,
                                    out_irrep=out_irrep,
                                    multiplicity=multiplicity,
                                    weight_offset=weight_offset,
                                )
                            )
                            reference_intermediate.append(
                                (multiplicity, node_edge_irrep)
                            )
                            expanded_output.append((multiplicity, out_irrep))
                            reference_node_edge_instructions.append(
                                (
                                    node_index,
                                    edge_index,
                                    path_index,
                                    "uvu",
                                    True,
                                    1.0,
                                )
                            )
                            reference_edge_edge_instructions.append(
                                (
                                    path_index,
                                    extra_index,
                                    path_index,
                                    "uvu",
                                    True,
                                    1.0,
                                )
                            )
                            weight_offset += multiplicity

        if not paths:
            raise ValueError("No Wigner-6j coupling paths were generated")

        self.irreps_out = o3.Irreps(expanded_output)

        reference_node_edge_tp = o3.TensorProduct(
            self.irreps_node_feats,
            self.irreps_edge_attrs,
            o3.Irreps(reference_intermediate),
            reference_node_edge_instructions,
            internal_weights=False,
            shared_weights=False,
        )
        reference_edge_edge_tp = o3.TensorProduct(
            o3.Irreps(reference_intermediate),
            self.extra_irreps_node_attrs,
            self.irreps_out,
            reference_edge_edge_instructions,
            internal_weights=False,
            shared_weights=False,
        )

        recoupled_intermediate = []
        recoupled_node_node_instructions = []
        recoupled_node_edge_instructions = []
        source_weight_indices = []
        recoupling_path_indices = []
        component_recoupling_coefficients = []

        for path_index, path in enumerate(paths):
            node_irrep = self.irreps_node_feats[path.node_index].ir
            edge_irrep = self.irreps_edge_attrs[path.edge_index].ir
            extra_irrep = self.extra_irreps_node_attrs[path.extra_index].ir
            for node_extra_irrep in node_irrep * extra_irrep:
                if path.out_irrep not in node_extra_irrep * edge_irrep:
                    continue

                coefficient = wigner_6j(
                    edge_irrep.l,
                    node_irrep.l,
                    path.node_edge_irrep.l,
                    extra_irrep.l,
                    path.out_irrep.l,
                    node_extra_irrep.l,
                )
                if abs(coefficient) < 1.0e-14:
                    continue

                intermediate_index = len(recoupled_intermediate)
                recoupled_intermediate.append((path.multiplicity, node_extra_irrep))
                recoupled_node_node_instructions.append(
                    (
                        path.node_index,
                        path.extra_index,
                        intermediate_index,
                        "uvu",
                        True,
                        1.0,
                    )
                )
                recoupled_node_edge_instructions.append(
                    (
                        intermediate_index,
                        path.edge_index,
                        path_index,
                        "uvu",
                        True,
                        1.0,
                    )
                )
                source_weight_indices.extend(
                    range(
                        path.weight_offset,
                        path.weight_offset + path.multiplicity,
                    )
                )
                recoupling_path_indices.append(path_index)
                component_recoupling_coefficients.append(coefficient)

        self.recoupled_node_node_tp = o3.TensorProduct(
            self.irreps_node_feats,
            self.extra_irreps_node_attrs,
            o3.Irreps(recoupled_intermediate),
            recoupled_node_node_instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.recoupled_node_edge_tp = O3ScatterTensorProduct(
            o3.Irreps(recoupled_intermediate),
            self.irreps_edge_attrs,
            self.irreps_out,
            instructions=recoupled_node_edge_instructions,
        )

        recoupling_coefficients = []
        for intermediate_index, (path_index, coefficient) in enumerate(
            zip(recoupling_path_indices, component_recoupling_coefficients)
        ):
            path = paths[path_index]
            node_extra_irrep = recoupled_intermediate[intermediate_index][1]
            reference_scale = (
                reference_node_edge_tp.instructions[path_index].path_weight
                * reference_edge_edge_tp.instructions[path_index].path_weight
            )
            recoupled_scale = (
                self.recoupled_node_node_tp.instructions[intermediate_index].path_weight
                * self.recoupled_node_edge_tp.tp.instructions[
                    intermediate_index
                ].path_weight
            )
            reference_component_scale = math.sqrt(
                path.node_edge_irrep.dim * path.out_irrep.dim
            )
            recoupled_component_scale = math.sqrt(
                node_extra_irrep.dim * path.out_irrep.dim
            )
            reference_element_scale = reference_scale / reference_component_scale
            recoupled_element_scale = recoupled_scale / recoupled_component_scale
            normalized_coefficient = (
                coefficient * reference_element_scale / recoupled_element_scale
            )
            recoupling_coefficients.extend(
                [normalized_coefficient] * paths[path_index].multiplicity
            )

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
            "source_weight_indices",
            torch.tensor(source_weight_indices, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "recoupling_coefficients",
            torch.tensor(recoupling_coefficients, dtype=torch.float64),
            persistent=False,
        )

    def _recoupled(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        extra_node_attrs: torch.Tensor,
        edge_weights: torch.Tensor,
        extra_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        indices = self.source_weight_indices
        coefficients = self.recoupling_coefficients.to(dtype=edge_weights.dtype)
        edge_weights = edge_weights.index_select(-1, indices)
        if self.weight_level == "edge":
            if extra_weights.size(0) != edge_index.size(1):
                raise ValueError("edge weights must have one row per graph edge")
            unit_weights = node_feats.new_ones(
                1,
                self.recoupled_node_node_tp.weight_numel,
            ).expand(node_feats.size(0), -1)
            node_node_intermediate = self.recoupled_node_node_tp(
                node_feats,
                extra_node_attrs,
                unit_weights,
            )
            edge_weights = edge_weights * extra_weights.index_select(
                -1,
                indices,
            )
        else:
            if extra_weights.size(0) != node_feats.size(0):
                raise ValueError("node weights must have one row per graph node")
            node_node_intermediate = self.recoupled_node_node_tp(
                node_feats,
                extra_node_attrs,
                extra_weights.index_select(-1, indices),
            )
        return self.recoupled_node_edge_tp(
            node_node_intermediate,
            edge_attrs,
            edge_weights * coefficients,
            edge_index,
        )

    def _reference_edges(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        extra_node_attrs: torch.Tensor,
        edge_weights: torch.Tensor,
        extra_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        node_edge_intermediate = self.reference_node_edge_tp(
            node_feats[source], edge_attrs, edge_weights
        )
        if self.weight_level == "edge":
            if extra_weights.size(0) != edge_index.size(1):
                raise ValueError("edge weights must have one row per graph edge")
            reference_extra_weights = extra_weights
        else:
            if extra_weights.size(0) != node_feats.size(0):
                raise ValueError("node weights must have one row per graph node")
            reference_extra_weights = extra_weights[source]
        return self.reference_edge_edge_tp(
            node_edge_intermediate,
            extra_node_attrs[source],
            reference_extra_weights,
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
        return self._recoupled(
            node_feats,
            edge_attrs,
            extra_node_attrs,
            edge_weights,
            extra_weights,
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
        """Evaluate the reference tree for numerical validation."""

        if not self.register_reference:
            raise RuntimeError(
                "forward_reference requires register_reference=True at construction"
            )

        messages = self._reference_edges(
            node_feats,
            edge_attrs,
            extra_node_attrs,
            edge_weights,
            extra_weights,
            edge_index,
        )
        return scatter_sum(
            messages,
            edge_index[1],
            dim=0,
            dim_size=node_feats.size(0),
        )
