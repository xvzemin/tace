################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Callable, NamedTuple, Optional, Sequence

import torch
from e3nn.math import normalize2mom

from .irreps import Irrep, Irreps, IrrepsLike
from .tensor_product import _quarter_turn


class Activation(torch.nn.Module):
    """Apply normalized scalar activations to O(2) features.

    Parameters
    ----------
    irreps_in : Irreps, str, or sequence
        Input representation. Non-``None`` activations are valid only for
        order-zero entries.
    acts : sequence of callable or None
        One activation per input entry. ``None`` leaves that entry unchanged.
        Each activation is rescaled to unit second moment for standard-normal
        inputs. Its parity determines the output reflection and time parity.

    Notes
    -----
    Inputs and outputs use flattened ``ir_mul`` layout with shape
    ``(..., irreps.dim)``.
    """

    def __init__(
        self,
        irreps_in: IrrepsLike,
        acts: Sequence[Optional[Callable]],
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
            if ir.is_invariant_scalar():
                irreps_out.append((ir, mul))
                continue
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
            if activation_parity == 0 and (ir.p == -1 or ir.t == -1):
                raise ValueError(
                    "An activation acting on a reflection- or time-reversal-odd "
                    "scalar must be either even or odd."
                )
            if activation_parity == -1:
                output_parity = ir.p
                output_time_parity = ir.t
            else:
                output_parity = 1
                output_time_parity = 1
            irreps_out.append((Irrep(0, output_parity, output_time_parity), mul))

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
            return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return features[..., :0]

    def __repr__(self) -> str:
        acts = "".join("x" if act is not None else " " for act in self.acts)
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"


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
    irreps_scalars : Irreps, str, or sequence
        Scalar entries transformed directly by ``act_scalars``.
    act_scalars : sequence of callable or None
        One normalized activation for each scalar entry.
    irreps_gates : Irreps, str, or sequence
        Order-zero entries used as gates. Their total multiplicity must equal
        the total multiplicity in ``irreps_gated``. Output irreps follow the
        tensor product of each activated gate and its gated entry.
    act_gates : sequence of callable or None
        One normalized activation for each gate entry.
    irreps_gated : Irreps, str, or sequence
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
        act_scalars: Sequence[Optional[Callable]],
        irreps_gates: IrrepsLike,
        act_gates: Sequence[Optional[Callable]],
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

        self.irreps_in, p, _ = (irreps_scalars + irreps_gates + irreps_gated).sort()
        scalar_end = len(irreps_scalars)
        gate_end = scalar_end + len(irreps_gates)
        self._scalar_locations = p[:scalar_end]
        self._gate_locations = p[scalar_end:gate_end]
        self._gated_locations = p[gate_end:]
        self._input_slices = self.irreps_in.slices()
        self._gate_slices = self.act_gates.irreps_out.slices()

        paths = []
        output_irrep_list = []
        i_gate = i_gated = 0
        gate_start = gated_start = 0
        while i_gate < len(self.act_gates.irreps_out):
            ir_gate, mul_gate = self.act_gates.irreps_out[i_gate]
            ir_gated, mul_gated = irreps_gated[i_gated]
            count = min(mul_gate - gate_start, mul_gated - gated_start)
            (ir_out,) = ir_gated * ir_gate
            paths.append(
                _GatePath(
                    i_gate,
                    gate_start,
                    i_gated,
                    gated_start,
                    count,
                    ir_gate,
                    ir_gated,
                    ir_out,
                )
            )
            output_irrep_list.append((ir_out, count))
            gate_start += count
            gated_start += count
            if gate_start == mul_gate:
                i_gate += 1
                gate_start = 0
            if gated_start == mul_gated:
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
            return features[..., :0]
        if len(locations) == 1:
            return features[..., self._input_slices[locations[0]]]
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
        outputs = [scalars]
        for path in self._paths:
            gate = gates[..., self._gate_slices[path.i_gate]][
                ..., path.gate_start : path.gate_start + path.mul
            ]
            i_in = self._gated_locations[path.i_gated]
            values = features[..., self._input_slices[i_in]].reshape(
                *features.shape[:-1],
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
