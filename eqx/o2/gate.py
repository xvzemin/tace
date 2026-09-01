################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import NamedTuple, Optional, Sequence

import torch
from e3nn.math import normalize2mom

from .irreps import Irrep, Irreps, IrrepsLike


def _parity_order(ir: Irrep) -> tuple[int, int]:
    return ir.m, {1: 0, -1: 1, 0: 2}[ir.p]


def _quarter_turn(features: torch.Tensor) -> torch.Tensor:
    return torch.stack((-features[..., 1, :], features[..., 0, :]), dim=-2)


class Activation(torch.nn.Module):
    """Apply normalized scalar activations to O(2) features.

    Parameters
    ----------
    irreps_in : IrrepsLike
        Input representation. Non-``None`` activations are valid only for
        order-zero entries.
    acts : sequence of torch.nn.Module or None
        One activation per input entry. ``None`` leaves that entry unchanged.
        Each activation is rescaled to preserve second moments, and its parity
        determines the corresponding output scalar parity.

    Notes
    -----
    Inputs and outputs use flattened ``ir_mul`` layout with shape
    ``(..., irreps.dim)``.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        acts: Sequence[Optional[torch.nn.Module]],
    ) -> None:
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        if len(self.irreps_in) != len(acts):
            raise ValueError(
                "Irreps and activation counts do not match: "
                f"{len(self.irreps_in)} != {len(acts)}."
            )

        normalized_acts = [
            normalize2mom(act) if act is not None else None for act in acts
        ]
        irreps_out = []
        for (ir, mul), act in zip(self.irreps_in, normalized_acts):
            if act is None:
                irreps_out.append((ir, mul))
                continue
            if ir.m != 0:
                raise ValueError("Activation functions can only act on scalars.")
            reference = next(act.parameters(), None)
            if reference is None:
                reference = next(act.buffers(), None)
            kwargs = {}
            if reference is not None:
                kwargs["device"] = reference.device
                if reference.is_floating_point():
                    kwargs["dtype"] = reference.dtype
            values = torch.linspace(0.0, 10.0, 256, **kwargs)
            training = act.training
            act.eval()
            try:
                with torch.no_grad():
                    positive = act(values)
                    negative = act(-values)
            finally:
                act.train(training)
            scale = torch.maximum(
                torch.ones((), device=positive.device, dtype=positive.dtype),
                torch.maximum(positive.abs().max(), negative.abs().max()),
            )
            even = (positive - negative).abs().max() <= 1.0e-5 * scale
            odd = (positive + negative).abs().max() <= 1.0e-5 * scale
            activation_parity = 1 if even else -1 if odd else 0
            output_parity = activation_parity if ir.p == -1 else ir.p
            if output_parity == 0:
                raise ValueError("Odd scalar activation must be either even or odd.")
            irreps_out.append((Irrep(0, output_parity), mul))

        self.irreps_out = Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(normalized_acts)
        self._slices = self.irreps_in.slices()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the configured activations.

        Parameters
        ----------
        features : torch.Tensor
            Input with shape ``(..., irreps_in.dim)``.

        Returns
        -------
        torch.Tensor
            Activated features with shape ``(..., irreps_out.dim)``.
        """
        if features.ndim < 1 or features.size(-1) != self.irreps_in.dim:
            raise ValueError(
                "Activation feature trailing dimension must be "
                f"{self.irreps_in.dim}, got {tuple(features.shape)}."
            )
        outputs = []
        for ir_slice, act in zip(self._slices, self.acts):
            values = features[..., ir_slice]
            outputs.append(values if act is None else act(values))
        if outputs:
            return torch.cat(outputs, dim=-1)
        return features.new_empty(*features.shape[:-1], 0)


class _GatePath(NamedTuple):
    i_gate: int
    gate_start: int
    i_gated: int
    gated_start: int
    mul: int
    ir_gate: Irrep
    ir_gated: Irrep
    ir_out: Irrep


class Gate(torch.nn.Module):
    """Apply scalar activations and scalar gates to O(2) features.

    Parameters
    ----------
    irreps_scalars : IrrepsLike
        Scalar entries transformed directly by ``act_scalars``.
    act_scalars : sequence of torch.nn.Module or None
        One normalized activation for each scalar entry.
    irreps_gates : IrrepsLike
        Order-zero entries used as gates. Their total multiplicity must equal
        the total multiplicity in ``irreps_gated``.
    act_gates : sequence of torch.nn.Module or None
        One normalized activation for each gate entry.
    irreps_gated : IrrepsLike
        Entries multiplied channel-wise by the activated gates.

    Notes
    -----
    ``irreps_in`` is the canonical ordering of scalar, gate, and gated entries.
    The output contains the activated scalars followed by gated entries. All
    tensors use flattened ``ir_mul`` layout.
    """

    def __init__(
        self,
        irreps_scalars: IrrepsLike,
        act_scalars: Sequence[Optional[torch.nn.Module]],
        irreps_gates: IrrepsLike,
        act_gates: Sequence[Optional[torch.nn.Module]],
        irreps_gated: IrrepsLike,
    ) -> None:
        super().__init__()
        irreps_scalars = Irreps(irreps_scalars)
        irreps_gates = Irreps(irreps_gates)
        irreps_gated = Irreps(irreps_gated)
        if any(ir.m != 0 for ir, _ in irreps_scalars):
            raise ValueError("Gate scalars must be scalar O(2) irreps.")
        if any(ir.m != 0 for ir, _ in irreps_gates):
            raise ValueError("Gate inputs must be scalar O(2) irreps.")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} gated irreps, but "
                f"{irreps_gates.num_irreps} gate scalars."
            )

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        self.act_gates = Activation(irreps_gates, act_gates)
        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated

        tagged_entries = (
            [
                ("scalar", index, mul, ir)
                for index, (ir, mul) in enumerate(irreps_scalars)
            ]
            + [("gate", index, mul, ir) for index, (ir, mul) in enumerate(irreps_gates)]
            + [
                ("gated", index, mul, ir)
                for index, (ir, mul) in enumerate(irreps_gated)
            ]
        )
        tagged_entries.sort(key=lambda item: _parity_order(item[3]))
        self.irreps_in = Irreps([(ir, mul) for _, _, mul, ir in tagged_entries])
        input_locations = {
            (kind, index): location
            for location, (kind, index, _, _) in enumerate(tagged_entries)
        }
        self._scalar_locations = tuple(
            input_locations[("scalar", index)] for index in range(len(irreps_scalars))
        )
        self._gate_locations = tuple(
            input_locations[("gate", index)] for index in range(len(irreps_gates))
        )
        self._gated_locations = tuple(
            input_locations[("gated", index)] for index in range(len(irreps_gated))
        )
        self._input_slices = self.irreps_in.slices()

        paths = []
        output_irrep_list = []
        i_gate = i_gated = 0
        gate_start = gated_start = 0
        while i_gate < len(self.act_gates.irreps_out):
            gate_ir, gate_mul = self.act_gates.irreps_out[i_gate]
            gated_ir, gated_mul = irreps_gated[i_gated]
            count = min(gate_mul - gate_start, gated_mul - gated_start)
            product = gated_ir * gate_ir
            if len(product) != 1:
                raise RuntimeError("A scalar gate must produce one O(2) irrep.")
            ir_out = product[0]
            paths.append(
                _GatePath(
                    i_gate,
                    gate_start,
                    i_gated,
                    gated_start,
                    count,
                    gate_ir,
                    gated_ir,
                    ir_out,
                )
            )
            output_irrep_list.append((ir_out, count))
            gate_start += count
            gated_start += count
            if gate_start == gate_mul:
                i_gate += 1
                gate_start = 0
            if gated_start == gated_mul:
                i_gated += 1
                gated_start = 0
        self._paths = tuple(paths)
        self.irreps_out = self.act_scalars.irreps_out + Irreps(output_irrep_list)

    def _select_entries(
        self,
        features: torch.Tensor,
        locations: tuple[int, ...],
    ) -> torch.Tensor:
        if not locations:
            return features.new_empty(*features.shape[:-1], 0)
        return torch.cat(
            [features[..., self._input_slices[index]] for index in locations],
            dim=-1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Activate scalar entries and apply the gates.

        Parameters
        ----------
        features : torch.Tensor
            Input with shape ``(..., irreps_in.dim)``.

        Returns
        -------
        torch.Tensor
            Gated output with shape ``(..., irreps_out.dim)``.
        """
        if features.ndim < 1 or features.size(-1) != self.irreps_in.dim:
            raise ValueError(
                "Gate feature trailing dimension must be "
                f"{self.irreps_in.dim}, got {tuple(features.shape)}."
            )
        scalars = self.act_scalars(
            self._select_entries(features, self._scalar_locations)
        )
        if not self._paths:
            return scalars
        gates = self.act_gates(self._select_entries(features, self._gate_locations))
        gated = self._select_entries(features, self._gated_locations)
        gate_slices = self.act_gates.irreps_out.slices()
        gated_slices = self.irreps_gated.slices()
        outputs = [scalars]
        for path in self._paths:
            gate = gates[..., gate_slices[path.i_gate]].reshape(
                *gates.shape[:-1],
                self.act_gates.irreps_out[path.i_gate].mul,
            )[..., path.gate_start : path.gate_start + path.mul]
            values = gated[..., gated_slices[path.i_gated]].reshape(
                *gated.shape[:-1],
                path.ir_gated.dim,
                self.irreps_gated[path.i_gated].mul,
            )[..., path.gated_start : path.gated_start + path.mul]
            if path.ir_gate.is_odd_scalar() and path.ir_gated.m > 0:
                values = _quarter_turn(values)
            output = values * gate.unsqueeze(-2)
            outputs.append(
                output.reshape(*features.shape[:-1], path.ir_out.dim * path.mul)
            )
        return torch.cat(outputs, dim=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"
