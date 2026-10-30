"""Normalized scalar activations and Cartesian O(2) gates."""

import torch

from eqx import o2

from .irreps import Irreps
from .tensor_product import quarter_turn


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
    backend : {"torch", "cuda"}, optional
        PyTorch operations or fused CUDA evaluation, including derivatives.

    Examples
    --------
    >>> from eqx import co2
    >>> gate = co2.Gate("2x0e", [torch.tanh], "3x0e", [torch.sigmoid], "3x2m")
    >>> x = co2.Irreps(gate.irreps_in).randn(5, -1)
    >>> gate(x).shape
    torch.Size([5, 14])
    """

    def __init__(
        self,
        irreps_scalars,
        act_scalars,
        irreps_gates,
        act_gates,
        irreps_gated,
        *,
        backend="torch",
    ):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be 'torch' or 'cuda'.")
        self.backend = backend
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
        self._scalar_locations = reference._scalar_locations
        self._gate_locations = reference._gate_locations
        self._gated_locations = reference._gated_locations
        self._paths = reference._paths
        self.sizes = tuple(entry.dim for entry in self.irreps_in)
        if backend == "cuda":
            from ..ops.gate import gate_program

            self.cuda_program = gate_program(self, cartesian=True)

    def forward(self, features):
        """Apply scalar activations and gate STF tensors.

        Parameters
        ----------
        features : torch.Tensor
            STF features of shape ``(..., irreps_in.dim)`` in ``mul_ir`` layout.

        Returns
        -------
        torch.Tensor
            STF features of shape ``(..., irreps_out.dim)``.
        """
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        if self.backend == "cuda" and features.is_cuda:
            from ..ops.gate import gate

            return gate(features, self.cuda_program, self.irreps_out.dim)
        inputs = features.split(self.sizes, -1)
        outputs = [
            inputs[i] if act is None else act(inputs[i])
            for i, act in zip(self._scalar_locations, self.act_scalars.acts)
        ]
        gates = [
            inputs[i] if act is None else act(inputs[i])
            for i, act in zip(self._gate_locations, self.act_gates.acts)
        ]
        for path in self._paths:
            ir = self.irreps_gated[path.i_gated]
            value = inputs[self._gated_locations[path.i_gated]].reshape(
                *features.shape[:-1], ir.mul, ir.ir.dim
            )
            value = value[..., path.gated_start : path.gated_start + path.mul, :]
            if path.ir_gate.is_odd_scalar() and path.ir_gated.m:
                value = quarter_turn(value)
            scalar = gates[path.i_gate][
                ..., path.gate_start : path.gate_start + path.mul
            ]
            outputs.append((value * scalar.unsqueeze(-1)).flatten(-2))
        return torch.cat(outputs, -1) if outputs else features[..., :0]

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}"

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "mul.weight"
        if key in state_dict and state_dict[key].numel() == 0:
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
