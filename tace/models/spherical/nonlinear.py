################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

import torch
from e3nn import o3
from e3nn.nn import Gate

from tace.utils.env import acceleration_enabled

from ..linear import e3nnLinear
from ..mlp import ACTIVATION


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

    gate_cls = Gate
    if acceleration_enabled("eqx", kernel="linear"):
        from functools import partial

        from equivariantx.o3 import Gate as EqxGate

        gate_cls = partial(EqxGate, backend="cuda")
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
        nonlinearity = gate_cls(
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

        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_in if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_in if ir.l > 0])
        irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_gated)
        nonlinearity = gate_cls(
            irreps_scalars=irreps_scalars,
            act_scalars=[
                act_0e if irrep.is_scalar() else act_0o for _, irrep in irreps_scalars
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
