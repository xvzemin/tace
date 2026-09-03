################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Union

import torch
from e3nn import o3
from e3nn.nn import Activation, Gate

from ..linear import e3nnElementLinear, e3nnLinear
from ..mlp import ACTIVATION
from .base import ReadOut


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
    ) -> torch.Tensor:
        if not self.last_layer:
            return self._apply_linear(self.linear1[0], x, node_attrs)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](self._apply_linear(linear, x, node_attrs))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
        return self._apply_linear(self.linear2[-1], x, node_attrs)


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
    ) -> torch.Tensor:
        if not self.last_layer:
            return self._apply_linear(self.linear1[0], x, node_attrs)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](self._apply_linear(linear, x, node_attrs))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
        return self._apply_linear(self.linear2[-1], x, node_attrs)


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
    ) -> torch.Tensor:
        if node_attrs is None:
            raise ValueError("element readout requires node_attrs")
        return linear(x, node_attrs)


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
    ) -> torch.Tensor:
        if node_attrs is None:
            raise ValueError("element readout requires node_attrs")
        return linear(x, node_attrs)


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
