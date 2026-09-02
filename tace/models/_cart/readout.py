################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, Union

import torch

from eqx import co3

from ..mlp import get_scaled_activation
from .base import ReadOut
from .linear import ElementLinear, Linear
from .nonlinear import O3Gate


def _mask_fidelity_channels(
    features: torch.Tensor,
    irreps,
    node_fidelity: Optional[torch.Tensor],
    num_fidelities: int,
) -> torch.Tensor:
    if num_fidelities == 1 or node_fidelity is None:
        return features
    irreps = co3.Irreps(irreps)
    outputs = []
    for (ir, mul), ir_slice in zip(irreps, irreps.slices()):
        if mul % num_fidelities:
            raise ValueError("Readout multiplicity must divide by num_fidelities.")
        per_fidelity = mul // num_fidelities
        values = features[..., ir_slice].reshape(
            features.size(0), ir.dim, num_fidelities, per_fidelity
        )
        mask = features.new_zeros(features.size(0), num_fidelities)
        mask[torch.arange(features.size(0), device=features.device), node_fidelity] = 1
        values = values * mask[:, None, :, None]
        outputs.append(values.reshape(features.size(0), ir.dim * mul))
    return torch.cat(outputs, dim=-1)


class ScalarReadOut(ReadOut):
    def _make_linear(self, irreps_in, irreps_out):
        return Linear(irreps_in, irreps_out, bias=self.use_bias)

    def _apply_linear(self, linear, features, node_attrs):
        return linear(features)

    def _setup(self) -> None:
        if self.layer != self.num_layers - 1:
            self.linears = torch.nn.ModuleList(
                [self._make_linear(self.irreps_in, self.irreps_out)]
            )
            self.activations = torch.nn.ModuleList()
            self.last_layer = False
            return
        ir_out, _ = self.irreps_out[0]
        hidden = [
            co3.Irreps([(ir_out, channel * self.num_fidelities)])
            for channel in self.hidden_channel
        ]
        sequence = [self.irreps_in, *hidden, self.irreps_out]
        self.linears = torch.nn.ModuleList(
            self._make_linear(ir_in, ir_out_entry)
            for ir_in, ir_out_entry in zip(sequence, sequence[1:])
        )
        self.activations = torch.nn.ModuleList(
            co3.Activation(
                irreps,
                [get_scaled_activation("silu" if ir_out.p == 1 else "tanh")],
            )
            for irreps in hidden
        )
        self.last_layer = True

    def forward(self, features, node_fidelity=None, node_attrs=None):
        if not self.last_layer:
            return self._apply_linear(self.linears[0], features, node_attrs)
        for index, linear in enumerate(self.linears[:-1]):
            features = self._apply_linear(linear, features, node_attrs)
            features = self.activations[index](features)
            features = _mask_fidelity_channels(
                features,
                self.activations[index].irreps_out,
                node_fidelity,
                self.num_fidelities,
            )
        return self._apply_linear(self.linears[-1], features, node_attrs)


class TensorReadOut(ScalarReadOut):
    def _setup(self) -> None:
        if self.layer != self.num_layers - 1:
            self.linears = torch.nn.ModuleList(
                [self._make_linear(self.irreps_in, self.irreps_out)]
            )
            self.activations = torch.nn.ModuleList()
            self.last_layer = False
            return
        ir_out, _ = self.irreps_out[0]
        hidden = [
            co3.Irreps([(ir_out, channel * self.num_fidelities)])
            for channel in self.hidden_channel
        ]
        self.linears = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        irreps_in = self.irreps_in
        for irreps_gated in hidden:
            gate_mul = irreps_gated.num_irreps
            gate = O3Gate(
                co3.Irreps([(co3.Irrep("0e"), gate_mul)]),
                [get_scaled_activation("sigmoid")],
                irreps_gated,
            )
            self.linears.append(self._make_linear(irreps_in, gate.irreps_in))
            self.activations.append(gate)
            irreps_in = gate.irreps_out
        self.linears.append(self._make_linear(irreps_in, self.irreps_out))
        self.last_layer = True

    def forward(self, features, node_fidelity=None, node_attrs=None):
        if not self.last_layer:
            return self._apply_linear(self.linears[0], features, node_attrs)
        for index, linear in enumerate(self.linears[:-1]):
            features = self.activations[index](
                self._apply_linear(linear, features, node_attrs)
            )
            features = _mask_fidelity_channels(
                features,
                self.activations[index].irreps_out,
                node_fidelity,
                self.num_fidelities,
            )
        return self._apply_linear(self.linears[-1], features, node_attrs)


class ElementScalarReadOut(ScalarReadOut):
    def _make_linear(self, irreps_in, irreps_out):
        return ElementLinear(
            irreps_in,
            irreps_out,
            bias=self.use_bias,
            num_elements=self.num_elements,
        )

    def _apply_linear(self, linear, features, node_attrs):
        if node_attrs is None:
            raise ValueError("Element readout requires node_attrs.")
        return linear(features, node_attrs)


class ElementTensorReadOut(TensorReadOut, ElementScalarReadOut):
    pass


def _build(
    cls,
    *,
    num_layers,
    hidden_channel,
    bias,
    num_elements,
    num_fidelities,
    use_alllayer,
    parity,
    irreps_in,
    irreps_out,
):
    modules = torch.nn.ModuleList(
        cls(
            layer=layer,
            num_layers=num_layers,
            hidden_channel=hidden_channel,
            bias=bias,
            num_elements=num_elements,
            num_fidelities=num_fidelities,
            parity=parity,
            irreps_in=irreps_in[layer],
            irreps_out=irreps_out,
        )
        for layer in range(num_layers)
    )
    return modules if use_alllayer else torch.nn.ModuleList([modules[-1]])


def build_scalar_readout(**kwargs):
    return _build(ScalarReadOut, **kwargs)


def build_tensor_readout(**kwargs):
    return _build(TensorReadOut, **kwargs)


def build_element_scalar_readout(**kwargs):
    return _build(ElementScalarReadOut, **kwargs)


def build_element_tensor_readout(**kwargs):
    return _build(ElementTensorReadOut, **kwargs)


__all__ = [
    "build_element_scalar_readout",
    "build_element_tensor_readout",
    "build_scalar_readout",
    "build_tensor_readout",
]
