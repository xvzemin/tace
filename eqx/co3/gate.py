"""Normalized scalar activations and gated Cartesian tensors."""

import torch
from e3nn.nn import Activation as ScalarActivation

from .irreps import Irreps
from .tensor_product import ElementwiseTensorProduct


class Activation(torch.nn.Module):
    """Apply normalized scalar activations.

    Parameters
    ----------
    irreps_in : Irreps or str
        Scalar input representations.
    acts : sequence of callable or None
        One activation per entry. Activations on odd scalars must have
        definite parity. None leaves an entry unchanged.
    """

    def __init__(self, irreps_in, acts):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        if any(ir.l for _, ir in self.irreps_in):
            raise ValueError("Activation accepts only scalar representations.")
        self.activation = ScalarActivation(self.irreps_in.spherical(), acts)
        self.irreps_out = Irreps(self.activation.irreps_out)

    def forward(self, features):
        """Activate scalar features of shape (..., irreps_in.dim)."""
        return self.activation(features)

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}"


class Gate(torch.nn.Module):
    """Activate scalars and multiply Cartesian tensors by scalar gates.

    Parameters
    ----------
    irreps_scalars : Irreps or str
        Scalar outputs before activation.
    act_scalars : sequence of callable or None
        One normalized activation per scalar entry.
    irreps_gates : Irreps or str
        Scalar gates, with one channel per gated irrep.
    act_gates : sequence of callable or None
        One normalized activation per gate entry.
    irreps_gated : Irreps or str
        Symmetric traceless tensors multiplied by the activated gates.
    """

    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
    ):
        super().__init__()
        self.act_scalars = Activation(irreps_scalars, act_scalars)
        self.act_gates = Activation(irreps_gates, act_gates)
        self.irreps_scalars = Irreps(irreps_scalars).simplify()
        self.irreps_gates = Irreps(irreps_gates).simplify()
        self.irreps_gated = Irreps(irreps_gated).simplify()
        self.mul = ElementwiseTensorProduct(
            self.irreps_gated, self.act_gates.irreps_out, project=False
        )
        unsorted = self.irreps_scalars + self.irreps_gates + self.irreps_gated
        sorted_irreps, permutation, _ = unsorted.sort()
        self.irreps_in = sorted_irreps.simplify()
        self.irreps_out = self.act_scalars.irreps_out + self.mul.irreps_out
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
        """Evaluate features of shape (..., irreps_in.dim)."""
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        scalars = self.act_scalars(features.index_select(-1, self.indices_scalars))
        if not self.irreps_gates.dim:
            return scalars
        gates = self.act_gates(features.index_select(-1, self.indices_gates))
        gated = self.mul(features.index_select(-1, self.indices_gated), gates)
        return torch.cat((scalars, gated), dim=-1)

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}"
