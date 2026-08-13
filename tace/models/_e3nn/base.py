################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import abc
from typing import Union

import torch
from e3nn import o3

from ..lammps import e3nnGhostExchangeMixin


def _to_possible_tp_irreps(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    parity: bool,
    lmax: Union[int, None] = None,
) -> o3.Irreps:
    lmax = irreps_in2.lmax if lmax is None else lmax
    irrep_set = set(
        ir3
        for _, ir1 in irreps_in1
        for _, ir2 in irreps_in2
        for ir3 in ir1 * ir2
        if ir3.l <= lmax and (parity or ir3.p == (-1) ** ir3.l)
    )
    return o3.Irreps(irrep_set).regroup()


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
        self.bias = bias
        self.Lmax = Lmax
        self.lmax = lmax
        self.avg_num_neighbors = avg_num_neighbors

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
        mmax: int,
        Lmax: int,
        lmax: int,
        correlation: list[int],
        num_channel: int,
        edge_feats_channel: int,
        target_irreps: list[str],
        num_radial_basis: int,
        radial_mlp: list[int],
        radial_bias: bool,
        irreps_in: o3.Irreps,
        scalar_act: str,
        tensor_act: str,
        edge_ace_hidden: Union[int, None],
        l1l2: Union[str, None] = None,
        scatter_norm: str = "avg_num_neighbors",
        bias: bool = True,
        nonlinear: Union[str, None] = None,
        edge_nonlinear: Union[str, None] = None,
        resnet_type: str = "BB",
        resnet_linear_type: str = "aware",
        use_first_resnet: bool = False,
        pre_norm_type: Union[str, None] = None,
        use_first_pre_norm: bool = False,
        use_so2_edge_ace: bool = False,
        stochastic_depth: float = 0.0,
        use_first_dropout: bool = False,
        parity: bool = False,
        num_head: Union[int, None] = None,
        use_graph_softmax: bool = False,
        node_wise_hidden: Union[int, None] = None,
        edge_wise_hidden: Union[int, None] = None,
        gate_m0: bool = True,
        use_radial_phase: bool = True,
        num_mag_radial_basis: int = 8,
        magnetic_irreps: o3.Irreps = None,
        mag_Lmax: int = 1,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.num_layers = num_layers
        self.mmax = mmax
        self.Lmax = Lmax
        self.lmax = lmax
        self.correlation = correlation[layer]
        self.num_radial_basis = num_radial_basis
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_feats_channel = edge_feats_channel
        self.l1l2 = l1l2
        self.num_elements = num_elements
        self.num_channel = num_channel
        self.target_irreps = target_irreps
        self.radial_mlp = radial_mlp
        self.radial_bias = radial_bias
        self.use_bias = bias
        self.scatter_norm = scatter_norm
        self.scalar_act = scalar_act
        self.tensor_act = tensor_act
        self.edge_ace_hidden = edge_ace_hidden or num_channel
        if self.scatter_norm == "no_cutoff_density":
            self.apply_density_cutoff = False
        else:
            self.apply_density_cutoff = True
        self.radial_layer_norm = False
        if self.edge_feats_channel != self.num_radial_basis:
            self.radial_layer_norm = True
        self.nonlinear_type = None
        self.nonlinear_act = None
        if nonlinear is not None:
            self.nonlinear_act, self.nonlinear_type = nonlinear.split("_")
        self.gate_m0 = gate_m0
        self.use_radial_phase = use_radial_phase

        self.use_first_resnet = use_first_resnet
        self.resnet_type = resnet_type
        self.use_so2_edge_ace = use_so2_edge_ace
        self.use_first_dropout = use_first_dropout
        self.resnet_linear_type = resnet_linear_type
        self.pre_norm_type = pre_norm_type
        self.use_first_pre_norm = use_first_pre_norm
        self.nonlinear = nonlinear
        self.edge_nonlinear = edge_nonlinear
        self.stochastic_depth_p = stochastic_depth
        self.parity = parity
        self.num_head = num_head or 1
        self.use_graph_softmax = use_graph_softmax
        self.node_wise_hidden = node_wise_hidden or num_channel
        self.edge_wise_hidden = edge_wise_hidden or num_channel
        self.num_mag_radial_basis = num_mag_radial_basis
        self.magnetic_irreps = magnetic_irreps
        self.mag_Lmax = mag_Lmax

        self.irreps_in = irreps_in
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax, p=-1)
        if self.correlation == 1:
            self.irrreps_tp_out = _to_possible_tp_irreps(
                self.irreps_in, self.irreps_sh, parity, lmax=Lmax
            )
        else:
            self.irrreps_tp_out = _to_possible_tp_irreps(
                self.irreps_in, self.irreps_sh, parity, lmax=lmax
            )
        self.irreps_out = (self.irrreps_tp_out * num_channel).regroup()

        if self.layer == num_layers - 1:
            self.irreps_sc = (o3.Irreps(target_irreps) * num_channel).regroup()
        else:
            self.irreps_sc = _to_possible_tp_irreps(
                self.irreps_in, self.irreps_sh, parity, lmax=Lmax
            )
            self.irreps_sc = (self.irreps_sc * num_channel).regroup()

        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def _uses_edge_density(self) -> bool:
        return self.scatter_norm in {"density", "no_cutoff_density"}

    def _normalize_messages(
        self,
        messages: torch.Tensor,
        density: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if self.scatter_norm is None:
            return messages
        if self.scatter_norm == "avg_num_neighbors":
            return messages / self.avg_num_neighbors
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
        num_expert: Union[int, None],
        num_channel_per_expert: Union[int, None],
        target_irreps: list[str],
        irreps_in: o3.Irreps,
        correlation: list[int],
        l1l2: Union[str, None],
        bias: bool,
        nonlinear: Union[str, None],
        stochastic_depth: float = 0.0,
        use_first_dropout: bool = False,
        parity: bool = False,
        use_shared_expert: bool = False,
        agnostic: bool = False,
    ) -> None:
        super().__init__()

        self.layer = layer
        self.num_layers = num_layers
        self.Lmax = Lmax
        self.lmax = lmax
        self.correlation = correlation[layer]
        self.num_channel = num_channel
        self.agnostic = agnostic
        self.use_shared_expert = use_shared_expert
        self.num_expert = num_expert or 1
        self.num_channel_per_expert = num_channel_per_expert or self.num_channel
        if self.agnostic:
            assert self.use_shared_expert == False
            assert self.num_expert == 1
        self.num_hidden_channel = self.num_expert * self.num_channel_per_expert
        self.target_irreps = target_irreps
        self.num_elements = num_elements
        self.l1l2 = l1l2
        self.use_bias = bias
        self.nonlinear_type = None
        self.nonlinear_act = None
        if nonlinear is not None:
            self.nonlinear_act, self.nonlinear_type = nonlinear.split("_")
        self.stochastic_depth_p = stochastic_depth
        self.use_first_dropout = use_first_dropout
        self.parity = parity
        self.last_layer = layer == num_layers - 1
        self.irreps_in = irreps_in
        self.irreps_hidden = o3.Irreps(
            [(self.num_hidden_channel, ir) for _, ir in self.irreps_in]
        )

        self.irreps_tp_out_list = []
        for nu in range(2, self.correlation + 1):
            if nu == self.correlation:
                if self.last_layer:
                    self.irreps_tp_out_list.append(
                        (self.target_irreps * self.num_hidden_channel).regroup()
                    )
                else:
                    self.irreps_tp_out_list.append(
                        (
                            _to_possible_tp_irreps(
                                self.irreps_hidden, self.irreps_hidden, parity, Lmax
                            )
                            * self.num_hidden_channel
                        ).regroup()
                    )
            else:
                self.irreps_tp_out_list.append(
                    _to_possible_tp_irreps(
                        self.irreps_hidden, self.irreps_hidden, parity, lmax
                    )
                )

        if self.correlation == 1:
            self.irreps_coefs_out = o3.Irreps(
                [
                    (self.num_hidden_channel, ir)
                    for _, ir in self.irreps_in
                    if ir.l <= Lmax
                ]
            )
        else:
            self.irreps_coefs_out = (
                _to_possible_tp_irreps(self.irreps_in, self.irreps_in, parity, Lmax)
                * self.num_hidden_channel
            ).regroup()

        if self.last_layer:
            self.irreps_coefs_out = (
                self.target_irreps * self.num_hidden_channel
            ).regroup()
        else:
            self.irreps_coefs_out = self.irreps_coefs_out

        self.irreps_out = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.irreps_coefs_out]
        )

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
        num_fidelities: int,
        parity: bool,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ) -> None:
        super().__init__()

        self.scalar_act = "tanh" if "0o" in irreps_out else "silu"
        self.tensor_act = "sigmoid"
        self.layer = layer
        self.num_layers = num_layers
        self.use_bias = bias
        self.num_fidelities = num_fidelities
        self.parity = parity

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = (irreps_out * num_fidelities).regroup()

        self.irreps_gates = [
            o3.Irreps([(c * self.num_fidelities, (0, 1))]) for c in hidden_channel
        ]
        self.irreps_hidden = [
            o3.Irreps([(c * self.num_fidelities, self.irreps_out[0].ir)])
            for c in hidden_channel
        ]
        self.l = self.irreps_out.lmax
        assert len(set(self.irreps_out.ls)) == 1

        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"scalar_act={self.scalar_act}, "
            f"tensor_act={self.tensor_act}, "
            f"num_fidelities={self.num_fidelities}"
        )
