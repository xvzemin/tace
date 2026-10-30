"""Normalized scalar activations and Cartesian O(2) gates."""

import torch

from eqx import o2

from .irreps import Irreps
from .tensor_product import ElementwiseTensorProduct


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
        self.sizes = tuple(entry.dim for entry in self.irreps_in)

    def forward(self, features):
        """Activate STF features with trailing size irreps_in.dim."""
        values = [
            value if act is None else act(value)
            for value, act in zip(features.split(self.sizes, -1), self.activation.acts)
        ]
        return torch.cat(values, -1) if values else features[..., :0]

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
        reference = o2.Gate(
            Irreps(irreps_scalars).circular(),
            act_scalars,
            Irreps(irreps_gates).circular(),
            act_gates,
            Irreps(irreps_gated).circular(),
        )
        self.irreps_in, self.irreps_out = map(
            Irreps, (reference.irreps_in, reference.irreps_out)
        )
        self.act_scalars = reference.act_scalars
        self.act_gates = reference.act_gates
        self.irreps_scalars = Irreps(reference.irreps_scalars)
        self.irreps_gates = Irreps(reference.irreps_gates)
        self.irreps_gated = Irreps(reference.irreps_gated)
        self.mul = ElementwiseTensorProduct(
            self.irreps_gated, Irreps(reference.act_gates.irreps_out)
        )
        entries = self.irreps_scalars + self.irreps_gates + self.irreps_gated
        sorted_irreps, permutation, _ = entries.sort()
        sections = sorted_irreps.slices()
        offset = 0
        for name, irreps in (
            ("scalars", self.irreps_scalars),
            ("gates", self.irreps_gates),
            ("gated", self.irreps_gated),
        ):
            indices = [
                j
                for i in range(offset, offset + len(irreps))
                for j in range(
                    sections[permutation[i]].start, sections[permutation[i]].stop
                )
            ]
            self.register_buffer(
                f"indices_{name}",
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            offset += len(irreps)

    def forward(self, features):
        """Gate STF features with trailing size irreps_in.dim."""
        scalars = self.act_scalars(features.index_select(-1, self.indices_scalars))
        if not self.irreps_gates.dim:
            return scalars
        gates = self.act_gates(features.index_select(-1, self.indices_gates))
        gated = self.mul(features.index_select(-1, self.indices_gated), gates)
        return torch.cat((scalars, gated), -1)

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}"
