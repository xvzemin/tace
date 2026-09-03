################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

import torch
from e3nn import o3
from e3nn.nn import Gate

from ..linear import e3nnLinear
from ..mlp import ACTIVATION


def _get_gate_layer(
    irreps_in,
    irreps_out,
    gate_m0: bool,
    scalar_act: Union[str, list[str], None] = None,
    tensor_act: Union[str, None] = None,
    bias: bool = True,
):
    if gate_m0:
        if scalar_act is None:
            act_0e_name = "sigmoid"
        elif isinstance(scalar_act, str):
            act_0e_name = scalar_act
        elif isinstance(scalar_act, list) and len(scalar_act) == 2:
            if not all(isinstance(name, str) for name in scalar_act):
                raise TypeError("scalar_act entries must be strings.")
            act_0e_name = scalar_act[0]
        else:
            raise TypeError(
                "scalar_act must be None, a string, or a list of two strings "
                "for 0e and 0o."
            )
        act_0e = ACTIVATION[act_0e_name]()

        irreps_gated = irreps_in
        irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_in)
        nonlinearity = Gate(
            irreps_scalars=o3.Irreps(),
            act_scalars=[],
            irreps_gates=irreps_gates,
            act_gates=[act_0e] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        linear_down_irreps_out = nonlinearity.irreps_in.simplify()
        linear_nonlinearity = e3nnLinear(irreps_in, irreps_out, bias=bias)
    else:
        if scalar_act is None:
            act_0e_name = "silu"
            act_0o_name = "tanh"
        elif isinstance(scalar_act, str):
            act_0e_name = scalar_act
            act_0o_name = "tanh"
        elif isinstance(scalar_act, list) and len(scalar_act) == 2:
            act_0e_name, act_0o_name = scalar_act
            if not isinstance(act_0e_name, str) or not isinstance(act_0o_name, str):
                raise TypeError("scalar_act entries must be strings.")
        else:
            raise TypeError(
                "scalar_act must be None, a string, or a list of two strings "
                "for 0e and 0o."
            )
        if tensor_act is None:
            tensor_act = "sigmoid"
        if not isinstance(tensor_act, str):
            raise TypeError("tensor_act must be None or a string for tensor gates.")
        act_0e = ACTIVATION[act_0e_name]()
        act_0o = ACTIVATION[act_0o_name]()
        act_tensor = ACTIVATION[tensor_act]()

        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in irreps_in if ir.l == 0]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in irreps_in if ir.l > 0]
        )
        irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_gated)
        nonlinearity = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[
                act_0e if irrep.is_scalar() else act_0o
                for _, irrep in irreps_scalars
            ],
            irreps_gates=irreps_gates,
            act_gates=[act_tensor] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        linear_down_irreps_out = nonlinearity.irreps_in.simplify()
        linear_nonlinearity = e3nnLinear(
            nonlinearity.irreps_out,
            irreps_out,
            bias=bias,
        )

    return nonlinearity, linear_nonlinearity, linear_down_irreps_out


def get_nonlinear_layer(
    nonlinear: Union[str, None],
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    gate_m0: bool,
    scalar_act: Union[str, list[str], None] = None,
    tensor_act: Union[str, None] = None,
    bias: bool = True,
):
    if nonlinear is None:
        return torch.nn.Identity(), torch.nn.Identity(), irreps_in
    elif nonlinear == "gate":
        return _get_gate_layer(
            irreps_in,
            irreps_out,
            gate_m0,
            scalar_act,
            tensor_act,
            bias,
        )
    else:
        return _get_gate_layer(
            irreps_in,
            irreps_out,
            gate_m0,
            scalar_act,
            tensor_act,
            bias,
        )

# class O3Norm(torch.nn.Module):
#     def __init__(
#         self,
#         irreps: o3.Irreps,
#         activation: torch.nn.Module,
#     ) -> None:
#         super().__init__()

#         self.irreps_in = o3.Irreps(irreps)
#         self.irreps_out = o3.Irreps(irreps)
#         self.norm_fn = o3.Norm(self.irreps_in, squared=True)
#         self.register_buffer(
#             "balance_degree_weight",
#             1.0 / torch.tensor([2*l+1 for l in self.irreps_in.ls]),
#             persistent=False,
#         )
#         self.activation = Activation(self.norm_fn.irreps_out.regroup(), [activation])
#         self.scalar_multiplier = ElementwiseTensorProduct(
#             irreps_in1=self.norm_fn.irreps_out,
#             irreps_in2=self.irreps_in,
#         )

#     def forward(self, x: torch.Tensor, y: Union[torch.Tensor, None] = None) -> torch.Tensor:
#         norm = self.norm_fn(x) * self.balance_degree_weight
#         norm = self.activation(norm)
#         if y is not None:
#             return self.scalar_multiplier(norm, y)
#         return self.scalar_multiplier(norm, x)


# class ChannelWiseO3NormGate(torch.nn.Module):
#     """Gate all O(3) components belonging to the same channel together."""

#     def __init__(
#         self,
#         irreps: o3.Irreps,
#         activation: torch.nn.Module,
#     ) -> None:
#         super().__init__()

#         self.irreps_in = o3.Irreps(irreps)
#         self.irreps_out = self.irreps_in
#         multiplicities = {mul for mul, _ in self.irreps_in}
#         if len(multiplicities) != 1:
#             raise ValueError(
#                 "ChannelWiseO3NormGate requires the same multiplicity for every "
#                 f"irrep, got {self.irreps_in}"
#             )

#         from ..layout import LayoutTransform2
#         self.reshape = LayoutTransform2(self.irreps_in)
#         self.activation = activation
#         self.register_buffer(
#             "balance_degree_weight",
#             torch.cat(
#                 [
#                     torch.full((ir.dim,), 1.0 / ir.dim) / len(self.irreps_in)
#                     for _, ir in self.irreps_in
#                 ]
#             ),
#             persistent=False,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.reshape(x)
#         channel_norm = torch.sum(
#             x.square() * self.balance_degree_weight,
#             dim=-1,
#             keepdim=True,
#         )
#         gate = self.activation(channel_norm)

#         return self.reshape.inverse(gate * x)


# class O3BilinearGate(torch.nn.Module):
#     """Based on https://github.com/SamsungDS/GGNN"""

#     def __init__(self, irreps: o3.Irreps, l1l2: Union[str, None] = ">=") -> None:
#         super().__init__()

#         irreps = o3.Irreps(irreps)
#         assert irreps[0].ir == o3.Irrep("0e")
#         non_scalar_irreps = irreps[1:]
#         multiplicities = {mul for mul, _ in irreps}
#         assert len(multiplicities) == 1

#         from .fused import uuuTensorProduct
#         self.tp = uuuTensorProduct(
#             non_scalar_irreps,
#             irreps,
#             irreps,
#             l1l2=l1l2,
#             trainable=True,
#         )
#         self.irreps_target = irreps
#         self.irreps_gate = o3.Irreps(f"{self.tp.weight_numel}x0e")
#         self.irreps_act = irreps[:1]
#         self.irreps_non_scalar = non_scalar_irreps
#         self.irreps_in = (
#             self.irreps_gate
#             + self.irreps_act
#             + o3.Irreps([(2 * mul, ir) for mul, ir in non_scalar_irreps])
#         ).simplify()
#         self.irreps_out = (self.irreps_act + self.tp.irreps_out).simplify()

#         self._gate_dim = self.irreps_gate.dim
#         self._act_dim = self.irreps_act.dim
#         self._non_scalar_slices = list(
#             o3.Irreps([(2 * mul, ir) for mul, ir in non_scalar_irreps]).slices()
#         )
#         self._non_scalar_muls = [mul for mul, _ in non_scalar_irreps]
#         self._non_scalar_dims = [ir.dim for _, ir in non_scalar_irreps]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         weights = torch.nn.functional.silu(x[:, : self._gate_dim])
#         scalars = torch.nn.functional.silu(x[:, self._gate_dim : self._gate_dim + self._act_dim])
#         non_scalars = x[:, self._gate_dim + self._act_dim :]

#         x1_fields = []
#         x2_fields = []
#         for tensor_slice, mul, ir_dim in zip(
#             self._non_scalar_slices,
#             self._non_scalar_muls,
#             self._non_scalar_dims,
#         ):
#             field = non_scalars[:, tensor_slice].reshape(
#                 x.shape[0], 2 * mul, ir_dim
#             )
#             x1_fields.append(field[:, :mul].reshape(x.shape[0], -1))
#             x2_fields.append(field[:, mul:].reshape(x.shape[0], -1))

#         x1 = torch.cat(x1_fields, dim=-1)
#         x2 = torch.cat([torch.ones_like(scalars), *x2_fields], dim=-1)
#         coupled = self.tp(x1, x2, weights)
#         return torch.cat([scalars, coupled], dim=-1)

#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}("
#             f"{self.irreps_in} -> {self.irreps_out})"
#         )
