################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Union

import torch
from e3nn import o3
from e3nn.nn import Activation, Gate
from torch.fx.experimental.symbolic_shapes import statically_known_true

from equivariantx.o3 import path_matrices

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


def _sparse_symmetric_blocks(values, rows, reverse, num_pairs):
    """Sum active unordered pairs once and mirror them into the dense output."""
    key = torch.minimum(rows, reverse)
    keys, order = key.sort()
    opposite = torch.maximum(rows, reverse).index_select(0, order)
    oriented = torch.where((rows == key)[:, None, None], values, values.mT)
    # Only the first slot in each equal-key run receives contributions. This
    # keeps shapes static even with repeated periodic images. Mirroring the
    # already-reduced block keeps H_ij == H_ji.T even for colliding GPU adds.
    slot = torch.searchsorted(keys, key)
    reduced = torch.zeros_like(values).index_add_(0, slot, oriented).mul_(0.5)
    blocks = values.new_zeros(num_pairs, 3, 3)
    blocks.index_add_(0, keys, reduced)
    return blocks.index_add_(0, opposite, reduced.mT)


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

    float64_buffers = ("basis",)

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
        self.component_dims = (1, 3, 5)
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
        basis = {path[-1].l: q for path, q in path_matrices(2)}
        self.register_buffer(
            "basis", torch.cat([basis[l] for l in range(3)], dim=1).reshape(3, 3, 9)
        )

    def to_cartesian(
        self, coefficients: torch.Tensor, fidelity: torch.Tensor | None
    ) -> torch.Tensor:
        """Select fidelity heads (optional for a single head) and convert blocks."""
        if fidelity is not None:
            parts = coefficients.split(
                [dim * self.num_fidelities for dim in self.component_dims], dim=-1
            )
            row = torch.arange(coefficients.size(0), device=coefficients.device)
            coefficients = torch.cat(
                [
                    part.reshape(coefficients.size(0), self.num_fidelities, dim)[
                        row, fidelity
                    ]
                    for part, dim in zip(parts, self.component_dims)
                ],
                dim=-1,
            )
        elif self.num_fidelities != 1:
            raise ValueError(
                "Multi-fidelity Cartesian conversion needs fidelity indices."
            )
        return torch.einsum(
            "pk,abk->pab",
            coefficients.to(torch.float64),
            self.basis.to(torch.float64),
        )

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
        sizes = counts.square()
        offsets = sizes.cumsum(0) - sizes
        node = torch.arange(features.size(0), device=features.device)
        start = ptr[batch]
        # In the packed dense output, pair (i, j) lives at row_base[i] + j.
        row_base = offsets[batch] + (node - start) * counts[batch] - start
        diagonal = row_base + node
        rows = torch.cat((diagonal, edge_pairs))
        coefficients = torch.cat(
            (self.node_readout(features), self.edge_readout(edge_feats))
        )
        origin = None if self.num_fidelities == 1 else source[edge_pairs]
        fidelity = (
            None
            if origin is None
            else torch.cat((node_fidelity, node_fidelity[origin]))
        )
        values = self.to_cartesian(coefficients, fidelity)
        # Switch only when an extra FP64 Hessian costs at least 72 MiB
        # and there are at least 64 dense rows per active contribution.
        # Unknown symbolic sizes, small cells and image-heavy batches retain
        # the dense transpose, avoiding a sort and any new shape guard.
        if (
            values.is_cuda
            and statically_known_true(source.numel() >= 1024**2)
            and statically_known_true(source.numel() >= 64 * rows.size(0))
        ):
            if origin is None:
                origin = source[edge_pairs]
            reverse_edges = row_base[target[edge_pairs]] + origin
            return _sparse_symmetric_blocks(
                values, rows, torch.cat((diagonal, reverse_edges)), source.numel()
            )
        blocks = values.new_zeros(source.numel(), 3, 3).index_add_(0, rows, values)
        if statically_known_true(ptr.numel() == 2):
            n = features.size(0)
            square = blocks.view(n, n, 3, 3)
            return (square + square.permute(1, 0, 3, 2)).mul_(0.5).reshape_as(blocks)
        reverse = row_base[target] + source
        return (blocks + blocks.index_select(0, reverse).transpose(-1, -2)).mul_(0.5)


def mh_mask(
    x: torch.Tensor,
    node_fidelity: torch.Tensor,
    num_fidelities: int,
) -> torch.Tensor:
    """Mask contiguous head blocks without duplicating the mask per channel."""
    nodes, width = x.shape
    mask = x.new_zeros((nodes, num_fidelities))
    mask[torch.arange(nodes, device=x.device), node_fidelity] = 1
    blocks = x.reshape(nodes, num_fidelities, width // num_fidelities)
    return (blocks * mask.unsqueeze(-1)).reshape_as(x)


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
        node_type: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if not self.last_layer:
            return self._apply_linear(self.linear1[0], x, node_type)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](self._apply_linear(linear, x, node_type))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities)
        return self._apply_linear(self.linear2[-1], x, node_type)


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
        node_type: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if not self.last_layer:
            return self._apply_linear(self.linear1[0], x, node_type)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](self._apply_linear(linear, x, node_type))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities)
        return self._apply_linear(self.linear2[-1], x, node_type)


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
        node_type: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if node_type is None:
            raise ValueError("element readout requires node_type")
        return linear(x, node_type)


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
        node_type: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if node_type is None:
            raise ValueError("element readout requires node_type")
        return linear(x, node_type)


def build_scalar_readout(
    num_layers: int,
    hidden_channel: List[int],
    bias: bool,
    num_elements: int,
    num_fidelities: int,
    use_alllayer: bool,
    parity: bool | None,
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
    parity: bool | None,
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
    parity: bool | None,
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
    parity: bool | None,
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
