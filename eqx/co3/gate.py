################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

import torch

from .irreps import Irrep, Irreps, IrrepsLike


def _check_odd_activation(activation: torch.nn.Module) -> None:
    reference = next(activation.parameters(), None)
    if reference is None:
        reference = next(activation.buffers(), None)
    kwargs = {}
    if reference is not None:
        kwargs["device"] = reference.device
        if reference.is_floating_point():
            kwargs["dtype"] = reference.dtype
    values = torch.linspace(0.0, 10.0, 256, **kwargs)
    training = activation.training
    activation.eval()
    try:
        with torch.no_grad():
            positive = activation(values)
            negative = activation(-values)
    finally:
        activation.train(training)
    scale = torch.maximum(
        torch.ones((), device=positive.device, dtype=positive.dtype),
        torch.maximum(positive.abs().max(), negative.abs().max()),
    )
    if (positive + negative).abs().max() > 1.0e-5 * scale:
        raise ValueError("act_0o must be an odd function.")


class Gate(torch.nn.Module):
    """Apply equivariant nonlinearities to complete Cartesian O(3) features.

    ``0e`` scalars use ``act_0e``. A provided ``act_0o`` must be odd. Every
    non-scalar irrep, and ``0o`` when ``act_0o`` is absent, is multiplied by
    an auxiliary ``0e`` gate activated by ``act_tensor``.
    """

    def __init__(
        self,
        irreps_out: IrrepsLike,
        *,
        act_0e: torch.nn.Module,
        act_0o: Union[torch.nn.Module, None],
        act_tensor: torch.nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(act_0e, torch.nn.Module):
            raise TypeError("act_0e must be a torch.nn.Module.")
        if act_0o is not None and not isinstance(act_0o, torch.nn.Module):
            raise TypeError("act_0o must be a torch.nn.Module or None.")
        if not isinstance(act_tensor, torch.nn.Module):
            raise TypeError("act_tensor must be a torch.nn.Module.")
        self.irreps_out = Irreps(irreps_out)
        direct_even = []
        direct_odd = []
        gated = []
        for multiplicity, irrep in self.irreps_out:
            if irrep == Irrep("0e"):
                direct_even.append((multiplicity, irrep))
            elif irrep == Irrep("0o") and act_0o is not None:
                direct_odd.append((multiplicity, irrep))
            else:
                gated.append((multiplicity, irrep))
        self.irreps_0e = Irreps(direct_even).regroup()
        self.irreps_0o = Irreps(direct_odd).regroup()
        self.irreps_gated = Irreps(gated)
        self.num_gates = self.irreps_gated.num_irreps
        self.irreps_gates = (
            Irreps([(self.num_gates, Irrep("0e"))])
            if self.num_gates
            else Irreps()
        )
        self.irreps_in = Irreps(
            tuple(self.irreps_0e)
            + tuple(self.irreps_gates)
            + tuple(self.irreps_0o)
            + tuple(self.irreps_gated)
        ).simplify()
        if act_0o is not None:
            _check_odd_activation(act_0o)
        self.act_0e = act_0e
        self.act_0o = act_0o
        self.act_tensor = act_tensor

        sources = [0] * self.irreps_out.dim
        output_offset = even_offset = odd_offset = gated_offset = 0
        odd_start = self.irreps_0e.dim
        gated_start = odd_start + self.irreps_0o.dim
        for multiplicity, irrep in self.irreps_out:
            width = multiplicity * irrep.dim
            if irrep == Irrep("0e"):
                source = even_offset
                even_offset += width
            elif irrep == Irrep("0o") and act_0o is not None:
                source = odd_start + odd_offset
                odd_offset += width
            else:
                source = gated_start + gated_offset
                gated_offset += width
            sources[output_offset : output_offset + width] = range(
                source, source + width
            )
            output_offset += width
        self.register_buffer(
            "_output_sources",
            torch.tensor(sources, dtype=torch.long),
            persistent=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            raise TypeError("Cartesian O(3) Gate supports real inputs only.")
        if input.ndim < 2 or input.shape[-2] != self.irreps_in.dim:
            raise ValueError(
                f"Gate input Cartesian dimension must be {self.irreps_in.dim}."
            )
        even_end = self.irreps_0e.dim
        gate_end = even_end + self.irreps_gates.dim
        odd_end = gate_end + self.irreps_0o.dim
        even = self.act_0e(input[..., :even_end, :])
        gates = self.act_tensor(input[..., even_end:gate_end, :])
        odd = input[..., gate_end:odd_end, :]
        if self.act_0o is not None:
            odd = self.act_0o(odd)
        tensors = input[..., odd_end:, :]
        if self.num_gates:
            expanded = []
            for gate_index, irrep in enumerate(self.irreps_gated.expanded()):
                expanded.append(
                    gates[..., gate_index : gate_index + 1, :].expand(
                        *gates.shape[:-2], irrep.dim, gates.shape[-1]
                    )
                )
            tensors = tensors * torch.cat(expanded, dim=-2)
        output = torch.cat((even, odd, tensors), dim=-2)
        return output.index_select(-2, self._output_sources)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out})"
