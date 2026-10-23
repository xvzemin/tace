################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import List, Union

import torch
from e3nn import o3
from e3nn.nn import Activation, Gate

from ..linear import e3nnElementLinear, e3nnLinear
from ..mlp import ACTIVATION
from .base import ReadOut
from .nonlinear import get_nonlinear_layer


class EdgeReadOut(torch.nn.Module):
    """Apply Linear--Gate--Linear to equivariant edge descriptors.

    Parameters
    ----------
    irreps_in, irreps_hidden, irreps_out : e3nn.o3.Irreps
        Input, post-gate and output representations.
    scalar_act : str, list of str or None, optional
        Scalar activation, or separate activations for even and odd scalars.
    tensor_act : str or None, optional
        Tensor-gate activation.
    bias : bool, optional
        Include scalar biases. Defaults to False.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_hidden: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        *,
        scalar_act: str | list[str] | None = None,
        tensor_act: str | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        (
            self.nonlinearity,
            self.linear_nonlinearity,
            irreps_gate,
        ) = get_nonlinear_layer(
            "gate",
            self.irreps_hidden,
            self.irreps_out,
            gate_m0=False,
            scalar_act=scalar_act,
            tensor_act=tensor_act,
            bias=bias,
        )
        self.linear_down = e3nnLinear(self.irreps_in, irreps_gate, bias=bias)

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """Return ``(num_edges, irreps_out.dim)`` without selecting output heads."""
        edge_feats = self.linear_down(edge_feats)
        return self.linear_nonlinearity(self.nonlinearity(edge_feats))


class DirectHessianReadout(torch.nn.Module):
    """Predict diagonal blocks from nodes and off-diagonal blocks from edges.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Node feature representations.
    irreps_edge : e3nn.o3.Irreps
        Edge feature representations.
    num_channel : int
        Hidden multiplicity in the edge readout.
    num_fidelities : int
        Number of output heads.
    scalar_act, tensor_act : str or None, optional
        Scalar and tensor-gate activations.
    bias : bool, optional
        Include a scalar bias in the diagonal readout.

    Notes
    -----
    Returns ``(sum_g N_g**2, 3, 3)`` in row-major atom-pair order.
    Enforces ``H_ij = H_ji.T`` but not energy-derivative consistency or
    the acoustic sum rule. Periodic image contributions are summed into
    fixed-cell atom-pair blocks.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_edge: o3.Irreps | str,
        num_channel: int,
        num_fidelities: int,
        *,
        scalar_act: str | list[str] | None = None,
        tensor_act: str | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.num_fidelities = num_fidelities
        self.irreps_out = o3.Irreps(
            f"{num_fidelities}x0e+{num_fidelities}x1e+{num_fidelities}x2e"
        )
        self.node_readout = e3nnLinear(self.irreps_in, self.irreps_out, bias=bias)
        self.edge_readout = EdgeReadOut(
            self.irreps_edge,
            o3.Irreps([(num_channel, ir) for _, ir in self.irreps_edge.regroup()]),
            self.irreps_out,
            scalar_act=scalar_act,
            tensor_act=tensor_act,
        )
        self.register_buffer(
            "basis",
            torch.cat(
                [
                    o3.wigner_3j(1, 1, ell, dtype=torch.float64) * sqrt(2 * ell + 1)
                    for ell in range(3)
                ],
                dim=-1,
            ),
        )

    def to_cartesian(
        self, coefficients: torch.Tensor, fidelity: torch.Tensor
    ) -> torch.Tensor:
        """Select fidelity heads and convert irreps to Cartesian blocks."""
        parts = coefficients.split(
            [self.num_fidelities, 3 * self.num_fidelities, 5 * self.num_fidelities],
            dim=-1,
        )
        coefficients = torch.cat(
            [
                part.reshape(coefficients.size(0), self.num_fidelities, dim)
                for part, dim in zip(parts, (1, 3, 5))
            ],
            dim=-1,
        )
        row = torch.arange(coefficients.size(0), device=coefficients.device)
        return torch.einsum("pk,abk->pab", coefficients[row, fidelity], self.basis)

    def forward(
        self,
        features: torch.Tensor,
        edge_feats: torch.Tensor,
        ptr: torch.Tensor,
        batch: torch.Tensor,
        node_fidelity: torch.Tensor,
        pair_index: torch.Tensor,
        edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        source, target = pair_index
        counts = ptr[1:] - ptr[:-1]
        offsets = counts.square().cumsum(0) - counts.square()
        node = torch.arange(features.size(0), device=features.device)
        diagonal = offsets[batch] + (node - ptr[batch]) * (counts[batch] + 1)
        blocks = features.new_zeros((source.numel(), 3, 3)).index_copy(
            0, diagonal, self.to_cartesian(self.node_readout(features), node_fidelity)
        )

        edge_index = pair_index[:, edge_pairs]
        coefficients = self.edge_readout(edge_feats)
        blocks = blocks.index_add(
            0, edge_pairs, self.to_cartesian(coefficients, node_fidelity[edge_index[0]])
        )

        graph = batch[source]
        reverse = (
            offsets[graph] + (target - ptr[graph]) * counts[graph] + source - ptr[graph]
        )
        return (blocks + blocks[reverse].transpose(-1, -2)) * 0.5


def mh_mask(
    x: torch.Tensor,
    node_fidelity: torch.Tensor,
    num_fidelities: int,
    l: int,
) -> torch.Tensor:
    B = x.size(0)
    fid_mul_ir = x.size(-1)
    ir = 2 * l + 1
    fid_mul = fid_mul_ir // ir
    mul = fid_mul // num_fidelities
    mask = torch.zeros(B, num_fidelities, mul, device=x.device, dtype=x.dtype)
    mask[torch.arange(B, device=x.device), node_fidelity, :] = 1
    mask = mask.reshape(B, fid_mul, 1)
    x = x.reshape(B, fid_mul, ir)
    return (x * mask).view(B, -1)


class ScalarReadOut(ReadOut):
    def _make_linear(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        return e3nnLinear(
            irreps_in,
            irreps_out,
            bias=self.use_bias,
        )

    def _apply_linear(
        self,
        linear: torch.nn.Module,
        x: torch.Tensor,
        node_attrs: Union[torch.Tensor, None],
        node_type: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        return linear(x)

    def _setup(self):

        if self.layer == self.num_layers - 1:
            self.linear2 = torch.nn.ModuleList()
            self.acts = torch.nn.ModuleList(
                Activation(irreps_in=hidden, acts=[ACTIVATION[self.scalar_act]()])
                for hidden in self.irreps_hidden
            )
            for irreps_in, irreps_out in zip(
                ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[:-1],
                ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[1:],
            ):
                self.linear2.append(self._make_linear(irreps_in, irreps_out))
            self.last_layer = True
        else:
            self.linear1 = torch.nn.ModuleList(
                [self._make_linear(self.irreps_in, self.irreps_out)]
            )
            self.last_layer = False

    def forward(
        self,
        x: torch.Tensor,
        node_fidelity: Union[torch.Tensor, None] = None,
        node_attrs: Union[torch.Tensor, None] = None,
        node_type: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if not self.last_layer:
            return self._apply_linear(self.linear1[0], x, node_attrs, node_type)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](self._apply_linear(linear, x, node_attrs, node_type))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
        return self._apply_linear(self.linear2[-1], x, node_attrs, node_type)


class TensorReadOut(ReadOut):
    def _make_linear(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        return e3nnLinear(
            irreps_in,
            irreps_out,
            bias=self.use_bias,
        )

    def _apply_linear(
        self,
        linear: torch.nn.Module,
        x: torch.Tensor,
        node_attrs: Union[torch.Tensor, None],
        node_type: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        return linear(x)

    def _setup(self):

        if self.layer == self.num_layers - 1:
            self.linear2 = torch.nn.ModuleList()
            self.acts = torch.nn.ModuleList()
            for irreps_gates, irreps_gated in zip(
                self.irreps_gates, self.irreps_hidden
            ):
                self.acts.append(
                    Gate(
                        irreps_scalars=o3.Irreps(),
                        act_scalars=[],
                        irreps_gates=irreps_gates,
                        act_gates=[ACTIVATION[self.tensor_act]()],
                        irreps_gated=irreps_gated,
                    )
                )
            for idx, (irreps_in, irreps_out) in enumerate(
                zip(
                    ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[:-2],
                    ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[1:-1],
                )
            ):
                self.linear2.append(
                    self._make_linear(irreps_in, self.acts[idx].irreps_in)
                )
            self.linear2.append(
                self._make_linear(self.irreps_hidden[-1], self.irreps_out)
            )
            self.last_layer = True
        else:
            self.linear1 = torch.nn.ModuleList(
                [self._make_linear(self.irreps_in, self.irreps_out)]
            )
            self.last_layer = False

    def forward(
        self,
        x: torch.Tensor,
        node_fidelity: Union[torch.Tensor, None] = None,
        node_attrs: Union[torch.Tensor, None] = None,
        node_type: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if not self.last_layer:
            return self._apply_linear(self.linear1[0], x, node_attrs, node_type)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](self._apply_linear(linear, x, node_attrs, node_type))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
        return self._apply_linear(self.linear2[-1], x, node_attrs, node_type)


class ElementScalarReadOut(ScalarReadOut):
    def _make_linear(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        return e3nnElementLinear(
            irreps_in,
            irreps_out,
            bias=self.use_bias,
            num_elements=self.num_elements,
        )

    def _apply_linear(
        self,
        linear: torch.nn.Module,
        x: torch.Tensor,
        node_attrs: Union[torch.Tensor, None],
        node_type: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if node_attrs is None:
            raise ValueError("element readout requires node_attrs")
        return linear(x, node_attrs, node_type)


class ElementTensorReadOut(TensorReadOut):
    def _make_linear(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        return e3nnElementLinear(
            irreps_in,
            irreps_out,
            bias=self.use_bias,
            num_elements=self.num_elements,
        )

    def _apply_linear(
        self,
        linear: torch.nn.Module,
        x: torch.Tensor,
        node_attrs: Union[torch.Tensor, None],
        node_type: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if node_attrs is None:
            raise ValueError("element readout requires node_attrs")
        return linear(x, node_attrs, node_type)


def build_scalar_readout(
    num_layers: int,
    hidden_channel: List[int],
    bias: bool,
    num_elements: int,
    num_fidelities: int,
    use_alllayer: bool,
    parity: bool,
    irreps_in: list[o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
):
    readouts = torch.nn.ModuleList()
    for layer in range(num_layers):
        readouts.append(
            ScalarReadOut(
                layer=layer,
                num_layers=num_layers,
                hidden_channel=hidden_channel,
                bias=bias,
                num_elements=num_elements,
                num_fidelities=num_fidelities,
                parity=parity,
                irreps_in=irreps_in[layer],
                irreps_out=o3.Irreps(irreps_out),
            )
        )
    if use_alllayer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])


def build_tensor_readout(
    num_layers: int,
    hidden_channel: int,
    bias: bool,
    num_elements: int,
    num_fidelities: int,
    use_alllayer: bool,
    parity: bool,
    irreps_in: list[o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
):
    readouts = torch.nn.ModuleList()
    for layer in range(num_layers):
        readouts.append(
            TensorReadOut(
                layer=layer,
                num_layers=num_layers,
                hidden_channel=hidden_channel,
                bias=bias,
                num_elements=num_elements,
                num_fidelities=num_fidelities,
                parity=parity,
                irreps_in=irreps_in[layer],
                irreps_out=o3.Irreps(irreps_out),
            )
        )
    if use_alllayer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])


def build_element_scalar_readout(
    num_layers: int,
    hidden_channel: List[int],
    bias: bool,
    num_fidelities: int,
    use_alllayer: bool,
    parity: bool,
    irreps_in: list[o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    num_elements: int,
):
    readouts = torch.nn.ModuleList(
        ElementScalarReadOut(
            layer=layer,
            num_layers=num_layers,
            hidden_channel=hidden_channel,
            bias=bias,
            num_fidelities=num_fidelities,
            parity=parity,
            irreps_in=irreps_in[layer],
            irreps_out=o3.Irreps(irreps_out),
            num_elements=num_elements,
        )
        for layer in range(num_layers)
    )
    return readouts if use_alllayer else torch.nn.ModuleList([readouts[-1]])


def build_element_tensor_readout(
    num_layers: int,
    hidden_channel: List[int],
    bias: bool,
    num_fidelities: int,
    use_alllayer: bool,
    parity: bool,
    irreps_in: list[o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    num_elements: int,
):
    readouts = torch.nn.ModuleList(
        ElementTensorReadOut(
            layer=layer,
            num_layers=num_layers,
            hidden_channel=hidden_channel,
            bias=bias,
            num_fidelities=num_fidelities,
            parity=parity,
            irreps_in=irreps_in[layer],
            irreps_out=o3.Irreps(irreps_out),
            num_elements=num_elements,
        )
        for layer in range(num_layers)
    )
    return readouts if use_alllayer else torch.nn.ModuleList([readouts[-1]])
