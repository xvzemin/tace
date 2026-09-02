################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, Union

import torch

from eqx import co3

from ..mlp import get_scaled_activation


def _activation(name: Optional[str], fallback: str) -> torch.nn.Module:
    return get_scaled_activation(name or fallback)


class Gate(torch.nn.Module):
    """Scalar activation and gating for flattened irreps."""

    def __init__(
        self,
        irreps_scalars,
        irreps_gated,
        *,
        scalar_act: Union[str, list[str], None] = None,
        tensor_act: Optional[str] = None,
    ) -> None:
        super().__init__()
        irreps_scalars = co3.Irreps(irreps_scalars)
        irreps_gated = co3.Irreps(irreps_gated)
        if isinstance(scalar_act, list):
            even_name = scalar_act[0] if scalar_act else "silu"
            odd_name = scalar_act[1] if len(scalar_act) > 1 else "tanh"
        else:
            even_name = scalar_act or "silu"
            odd_name = "tanh"
        scalar_acts = [
            _activation(even_name if ir.p == 1 else odd_name, "silu")
            for ir, _ in irreps_scalars
        ]
        gate_mul = irreps_gated.num_irreps
        irreps_gates = (
            co3.Irreps([(co3.Irrep("0e"), gate_mul)]) if gate_mul else co3.Irreps()
        )
        gate_acts = [_activation(tensor_act, "sigmoid")] if gate_mul else []
        self.gate = co3.Gate(
            irreps_scalars,
            scalar_acts,
            irreps_gates,
            gate_acts,
            irreps_gated,
        )
        self.irreps_in = self.gate.irreps_in
        self.irreps_out = self.gate.irreps_out

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.gate(features)


class O3Gate(torch.nn.Module):
    """Gate-only form used by tensor readouts."""

    def __init__(self, irreps_gates, act_gates, irreps_gated) -> None:
        super().__init__()
        self.gate = co3.Gate(
            co3.Irreps(),
            [],
            irreps_gates,
            act_gates,
            irreps_gated,
        )
        self.irreps_in = self.gate.irreps_in
        self.irreps_out = self.gate.irreps_out

    def forward(self, features, gates=None):
        if gates is not None:
            features = torch.cat((gates, features), dim=-1)
        return self.gate(features)


def split_scalars(irreps) -> tuple[co3.Irreps, co3.Irreps]:
    irreps = co3.Irreps(irreps)
    scalars = co3.Irreps([(ir, mul) for ir, mul in irreps if ir.l == 0])
    tensors = co3.Irreps([(ir, mul) for ir, mul in irreps if ir.l > 0])
    return scalars, tensors


__all__ = ["Gate", "O3Gate", "split_scalars"]
