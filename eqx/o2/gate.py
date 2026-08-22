################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Sequence, Union

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

    input = torch.linspace(0.0, 10.0, 256, **kwargs)
    training = activation.training
    activation.eval()
    try:
        with torch.no_grad():
            positive = activation(input)
            negative = activation(-input)
    finally:
        activation.train(training)

    scale = torch.maximum(
        torch.ones((), device=positive.device, dtype=positive.dtype),
        torch.maximum(positive.abs().max(), negative.abs().max()),
    )
    if (positive + negative).abs().max() > 1.0e-5 * scale:
        raise ValueError(
            "act_0o must be an odd function satisfying act_0o(-x) = -act_0o(x)."
        )


class O2Gate(torch.nn.Module):
    """Apply equivariant nonlinearities to complete O(2) features.

    ``0e`` outputs are passed directly through ``act_0e``. If ``act_0o`` is
    provided, it must be odd and is applied directly to ``0o`` outputs. If
    ``act_0o=None``, every ``0o`` output is instead multiplied by an
    auxiliary ``0e`` gate. Positive-order ``lm`` outputs are always
    multiplied by auxiliary ``0e`` gates. All auxiliary gates are passed
    through ``act_lm`` before multiplication.

    The input irreps are ordered as the output ``0e`` scalars, the auxiliary
    ``0e`` gates, the directly activated ``0o`` scalars, and the gated
    irreps. The output is restored to ``irreps_out`` order.

    An odd ``act_0o`` commutes with the sign change of ``0o`` under
    reflections, making this a general O(2)-equivariant operation. When used
    through :class:`O3O2Layout`, it also preserves global O(3) equivariance.
    """

    def __init__(
        self,
        irreps_out: IrrepsLike,
        *,
        act_0e: torch.nn.Module,
        act_0o: Union[torch.nn.Module, None],
        act_lm: torch.nn.Module,
    ) -> None:
        super().__init__()

        if act_0o is not None and not isinstance(act_0o, torch.nn.Module):
            raise TypeError("act_0o must be a torch.nn.Module or None.")

        self.irreps_out = Irreps(irreps_out)
        even_groups = []
        odd_groups = []
        gated_groups = []
        for multiplicity, irrep in self.irreps_out:
            if irrep == Irrep("0e"):
                even_groups.append((multiplicity, irrep))
            elif irrep == Irrep("0o") and act_0o is not None:
                odd_groups.append((multiplicity, irrep))
            else:
                gated_groups.append((multiplicity, irrep))

        self.irreps_0e = Irreps(even_groups).regroup()
        self.irreps_0o = Irreps(odd_groups).regroup()
        self.irreps_gated = Irreps(gated_groups)
        self.num_gates = self.irreps_gated.num_irreps
        self.irreps_gates = (
            Irreps([(self.num_gates, Irrep("0e"))]) if self.num_gates > 0 else Irreps()
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
        self.act_lm = act_lm

        output_sources = [0] * self.irreps_out.dim
        output_offset = 0
        even_offset = 0
        odd_offset = 0
        gated_offset = 0
        odd_start = self.irreps_0e.dim
        gated_start = odd_start + self.irreps_0o.dim
        for multiplicity, irrep in self.irreps_out:
            width = multiplicity * irrep.dim
            if irrep == Irrep("0e"):
                source_offset = even_offset
                even_offset += width
            elif irrep == Irrep("0o") and act_0o is not None:
                source_offset = odd_start + odd_offset
                odd_offset += width
            else:
                source_offset = gated_start + gated_offset
                gated_offset += width
            output_sources[output_offset : output_offset + width] = range(
                source_offset,
                source_offset + width,
            )
            output_offset += width

        self.register_buffer(
            "_output_sources",
            torch.tensor(output_sources, dtype=torch.long),
            persistent=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the gate to ``(..., irreps_in.dim, channels)`` input."""
        if input.is_complex():
            raise TypeError("O2Gate supports real inputs only.")
        if input.ndim < 2 or input.shape[-2] != self.irreps_in.dim:
            raise ValueError(
                "O2Gate input O(2) dimension must be "
                f"{self.irreps_in.dim}, got {tuple(input.shape)}."
            )

        even_end = self.irreps_0e.dim
        gate_end = even_end + self.irreps_gates.dim
        odd_end = gate_end + self.irreps_0o.dim
        even = self.act_0e(input[..., :even_end, :])
        gates = self.act_lm(input[..., even_end:gate_end, :])
        odd = input[..., gate_end:odd_end, :]
        if self.act_0o is not None:
            odd = self.act_0o(odd)
        gated = input[..., odd_end:, :]

        if self.num_gates > 0:
            gate_components = []
            for gate_index, irrep in enumerate(self.irreps_gated.expanded()):
                gate_components.append(
                    gates[..., gate_index : gate_index + 1, :].expand(
                        *gates.shape[:-2],
                        irrep.dim,
                        gates.shape[-1],
                    )
                )
            gated = gated * torch.cat(gate_components, dim=-2)

        output = torch.cat((even, odd, gated), dim=-2)
        return output.index_select(-2, self._output_sources)

    def forward_grouped(
        self,
        input_blocks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Apply the gate to grouped ``(..., irrep.dim, mul * channels)`` blocks."""
        if len(input_blocks) != len(self.irreps_in):
            raise ValueError("Expected one input block per O2Gate input group.")
        if not input_blocks:
            return ()
        if self.irreps_out != self.irreps_out.regroup():
            raise ValueError("Grouped O2Gate requires regrouped output irreps.")

        first_multiplicity = self.irreps_in[0][0]
        if input_blocks[0].shape[-1] % first_multiplicity != 0:
            raise ValueError("Invalid grouped O2Gate channel width.")
        channels = input_blocks[0].shape[-1] // first_multiplicity
        input_by_irrep = {
            irrep: block for (_, irrep), block in zip(self.irreps_in, input_blocks)
        }
        even_block = input_by_irrep.get(Irrep("0e"))
        even_width = self.irreps_0e.num_irreps * channels
        gate_width = self.num_gates * channels
        if even_width + gate_width > 0:
            if even_block is None or even_block.shape[-1] != even_width + gate_width:
                raise ValueError("Invalid grouped O2Gate 0e block width.")
        else:
            even_block = input_blocks[0].new_empty(
                *input_blocks[0].shape[:-2],
                1,
                0,
            )

        even_output = self.act_0e(even_block[..., :even_width])
        gate_block = self.act_lm(even_block[..., even_width:])
        outputs = []
        gate_offset = 0
        for multiplicity, irrep in self.irreps_out:
            width = multiplicity * channels
            if irrep == Irrep("0e"):
                outputs.append(even_output)
            elif irrep == Irrep("0o") and self.act_0o is not None:
                outputs.append(self.act_0o(input_by_irrep[irrep]))
            else:
                input_block = input_by_irrep[irrep]
                gate = gate_block[..., gate_offset : gate_offset + width]
                outputs.append(input_block * gate)
                gate_offset += width
        return tuple(outputs)

    def extra_repr(self) -> str:
        return f"{self.irreps_in} -> {self.irreps_out}"
