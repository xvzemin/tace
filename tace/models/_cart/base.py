################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import abc
from typing import Optional

import torch

from eqx import co3

from ..lammps import e3nnGhostExchangeMixin


def natural_irreps(lmax: int, multiplicity: int = 1) -> co3.Irreps:
    """Return natural-parity irreps through ``lmax``."""
    return co3.Irreps(
        [(co3.Irrep(l, (-1) ** l), multiplicity) for l in range(lmax + 1)]
    )


def possible_irreps(
    irreps_in1,
    irreps_in2,
    *,
    parity: bool,
    lmax: int,
) -> co3.Irreps:
    """Return unique output types allowed by two representations."""
    irreps_in1 = co3.Irreps(irreps_in1)
    irreps_in2 = co3.Irreps(irreps_in2)
    irrep_set = {
        ir_out
        for ir1, _ in irreps_in1
        for ir2, _ in irreps_in2
        for ir_out in ir1 * ir2
        if ir_out.l <= lmax and (parity or ir_out.p == (-1) ** ir_out.l)
    }
    return co3.Irreps([(ir, 1) for ir in irrep_set]).regroup()


class NodeEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        Lmax: int,
        lmax: int,
        avg_num_neighbors: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.Lmax = Lmax
        self.lmax = lmax
        self.avg_num_neighbors = avg_num_neighbors
        self.bias = bias
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError


class EdgeEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.bias = bias
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError


class EdgeUpdate(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_elements: int,
        num_radial_basis: int,
        num_channel: int,
        edge_embedding_channel: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.first_layer = layer == 0
        self.last_layer = layer == num_layers - 1
        self.num_elements = num_elements
        self.num_radial_basis = num_radial_basis
        self.num_channel = num_channel
        self.edge_embedding_channel = edge_embedding_channel
        self.use_bias = bias
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError


class Interaction(torch.nn.Module, e3nnGhostExchangeMixin):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_elements: int,
        avg_num_neighbors: float,
        Lmax: int,
        lmax: int,
        correlation: list[int],
        num_channel: int,
        edge_feats_channel: int,
        target_irreps,
        num_radial_basis: int,
        radial_mlp: list[int],
        radial_bias: bool,
        irreps_in,
        scalar_act=None,
        tensor_act: Optional[str] = None,
        edge_ace_hidden: Optional[int] = None,
        l1l2: Optional[str] = None,
        scatter_norm: Optional[str] = "avg_num_neighbors",
        bias: bool = True,
        nonlinear: Optional[str] = None,
        edge_nonlinear: Optional[str] = None,
        resnet_type: str = "BB",
        resnet_linear_type: str = "aware",
        use_first_resnet: bool = False,
        pre_norm_type: Optional[str] = None,
        use_first_pre_norm: bool = False,
        stochastic_depth: float = 0.0,
        use_first_dropout: bool = False,
        parity: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.num_elements = num_elements
        self.avg_num_neighbors = avg_num_neighbors
        self.Lmax = Lmax
        self.lmax = lmax
        self.correlation = correlation[layer]
        self.num_channel = num_channel
        self.edge_feats_channel = edge_feats_channel
        self.target_irreps = co3.Irreps(target_irreps)
        self.num_radial_basis = num_radial_basis
        self.radial_mlp = radial_mlp
        self.radial_bias = radial_bias
        self.scalar_act = scalar_act
        self.tensor_act = tensor_act
        self.edge_ace_hidden = edge_ace_hidden or num_channel
        self.l1l2 = l1l2
        self.scatter_norm = scatter_norm
        self.apply_density_cutoff = scatter_norm != "no_cutoff_density"
        self.use_bias = bias
        self.nonlinear = nonlinear
        self.edge_nonlinear = edge_nonlinear
        self.resnet_type = resnet_type
        self.resnet_linear_type = resnet_linear_type
        self.use_first_resnet = use_first_resnet
        self.pre_norm_type = pre_norm_type
        self.use_first_pre_norm = use_first_pre_norm
        self.stochastic_depth_p = stochastic_depth
        self.use_first_dropout = use_first_dropout
        self.parity = parity
        self.gate_m0 = kwargs.get("gate_m0", True)
        self.irreps_in = co3.Irreps(irreps_in)
        self.irreps_edge = natural_irreps(lmax)
        output_lmax = Lmax if self.correlation == 1 else lmax
        output_types = possible_irreps(
            self.irreps_in,
            self.irreps_edge,
            parity=parity,
            lmax=output_lmax,
        )
        self.irreps_out = co3.Irreps([(ir, num_channel) for ir, _ in output_types])
        if layer == num_layers - 1:
            self.irreps_sc = co3.Irreps(
                [(ir, num_channel) for ir, _ in self.target_irreps]
            )
        else:
            sc_types = possible_irreps(
                self.irreps_in,
                self.irreps_edge,
                parity=parity,
                lmax=Lmax,
            )
            self.irreps_sc = co3.Irreps([(ir, num_channel) for ir, _ in sc_types])
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def _uses_edge_density(self) -> bool:
        return self.scatter_norm in {"density", "no_cutoff_density"}

    def _normalize_messages(self, messages, density):
        if self.scatter_norm is None:
            return messages
        if self.scatter_norm == "avg_num_neighbors":
            return messages / self.avg_num_neighbors
        if self.scatter_norm == "sqrt_avg_num_neighbors":
            return messages / self.avg_num_neighbors**0.5
        return messages / density


class Product(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_elements: int,
        Lmax: int,
        lmax: int,
        num_channel: int,
        target_irreps,
        irreps_in,
        correlation: list[int],
        bias: bool,
        nonlinear=None,
        scalar_act=None,
        stochastic_depth: float = 0.0,
        use_first_dropout: bool = False,
        parity: bool = False,
        num_expert=None,
        num_channel_per_expert=None,
        use_shared_expert: bool = False,
        agnostic: bool = False,
        l1l2=None,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.num_elements = num_elements
        self.Lmax = Lmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.target_irreps = co3.Irreps(target_irreps)
        self.irreps_in = co3.Irreps(irreps_in)
        self.correlation = correlation[layer]
        self.use_bias = bias
        self.nonlinear = nonlinear
        self.scalar_act = scalar_act or "silu"
        self.stochastic_depth_p = stochastic_depth
        self.use_first_dropout = use_first_dropout
        self.parity = parity
        self.num_expert = num_expert or 1
        self.num_channel_per_expert = num_channel_per_expert or num_channel
        self.num_hidden_channel = self.num_expert * self.num_channel_per_expert
        self.use_shared_expert = use_shared_expert
        self.agnostic = agnostic
        self.l1l2 = l1l2
        if layer == num_layers - 1:
            output_types = self.target_irreps
        else:
            output_types = possible_irreps(
                self.irreps_in,
                self.irreps_in,
                parity=parity,
                lmax=Lmax,
            )
        self.irreps_out = co3.Irreps([(ir, num_channel) for ir, _ in output_types])
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError


class ReadOut(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        hidden_channel: list[int],
        bias: bool,
        num_elements: int,
        num_fidelities: int,
        parity: bool,
        irreps_in,
        irreps_out,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_channel = hidden_channel
        self.use_bias = bias
        self.num_elements = num_elements
        self.num_fidelities = num_fidelities
        self.parity = parity
        self.irreps_in = co3.Irreps(irreps_in)
        output_type = co3.Irreps(irreps_out)
        if len(output_type) != 1:
            raise ValueError("Readouts require exactly one output irrep.")
        ir, mul = output_type[0]
        self.irreps_out = co3.Irreps([(ir, mul * num_fidelities)])
        self.l = ir.l
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError
