"""Normalized scalar activations and Cartesian O(2) gates."""

import torch

from .. import o2
from .basis import ChangeOfBasis
from .irreps import Irreps


class Activation(torch.nn.Module):
    """Apply normalized scalar activations in Cartesian storage.

    Parameters
    ----------
    irreps_in : Irreps or str
        Input representations. Non-scalars require a None activation.
    acts : sequence of callable or None
        One activation per entry, respecting reflection and time parity.
    """

    def __init__(self, irreps_in, acts):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.activation = o2.Activation(self.irreps_in.circular(), acts)
        self.irreps_out = Irreps(self.activation.irreps_out)
        self.to_circular = ChangeOfBasis(self.irreps_in, inverse=True)
        self.to_cartesian = ChangeOfBasis(self.irreps_out)

    def forward(self, features):
        """Activate STF features with trailing size irreps_in.dim."""
        return self.to_cartesian(self.activation(self.to_circular(features)))

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}"


class Gate(torch.nn.Module):
    """Apply scalar activations and scalar-tensor gates.

    Parameters
    ----------
    irreps_scalars, irreps_gates : Irreps or str
        Scalar features and scalar gates before activation.
    act_scalars, act_gates : sequence of callable or None
        Normalized activations for each scalar and gate entry.
    irreps_gated : Irreps or str
        STF representations multiplied by the activated gates.
    """

    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
    ):
        super().__init__()
        self.gate = o2.Gate(
            Irreps(irreps_scalars).circular(),
            act_scalars,
            Irreps(irreps_gates).circular(),
            act_gates,
            Irreps(irreps_gated).circular(),
        )
        self.irreps_in, self.irreps_out = map(
            Irreps, (self.gate.irreps_in, self.gate.irreps_out)
        )
        self.to_circular = ChangeOfBasis(self.irreps_in, inverse=True)
        self.to_cartesian = ChangeOfBasis(self.irreps_out)

    def forward(self, features):
        """Gate STF features with trailing size irreps_in.dim."""
        return self.to_cartesian(self.gate(self.to_circular(features)))

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}"
