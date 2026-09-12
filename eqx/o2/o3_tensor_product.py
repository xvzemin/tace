################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Optional, Sequence

import torch
from e3nn import o3
from torch_geometric.utils import scatter

from .local_frame import LocalFrame


def _entry_components(frame: LocalFrame, index: int):
    """Locate signed magnetic components in regrouped local storage."""
    entry = frame._entries[index]
    slices = frame.local_irreps.slices()
    components = {}
    for m, (local_index, channel_slice) in enumerate(
        zip(entry.local_indices, entry.local_slices)
    ):
        _, mul = frame.local_irreps[local_index]
        start = slices[local_index].start + channel_slice.start
        if m == 0:
            components[0] = (start, 1.0)
        elif entry.odd:
            components[m] = (start + mul, 1.0)
            components[-m] = (start, -1.0)
        else:
            components[m] = (start, 1.0)
            components[-m] = (start + mul, 1.0)
    return components


class O3TensorProduct(torch.nn.Module):
    r"""Couple O(3) features to spherical harmonics in an edge-aligned frame.

    Parameters
    ----------
    irreps_in1 : o3.Irreps or str
        Feature representation, stored in flattened ``ir_mul`` order.
    irreps_in2 : o3.Irreps or str
        Edge spherical-harmonic representation. Each entry must have one
        channel, natural spatial parity, and even time parity.
    irreps_out : o3.Irreps or str
        Output representation, stored in flattened ``ir_mul`` order.
    instructions : sequence of tuple
        Paths ``(i_in1, i_in2, i_out, mode, train[, path_weight])``.
        Supports channelwise ``"uvu"`` and channel-mixing ``"uvw"`` paths.
    in1_var, in2_var, out_var : sequence of float, optional
        Input and output variances used in path normalization.
    irrep_normalization : {"component", "norm", "none"}, optional
        Normalization of the irreducible coupling maps.
    path_normalization : {"element", "path", "none"}, optional
        Normalization across paths reaching the same output entry.
    internal_weights : bool, optional
        Store trainable weights internally. The default is inferred from
        ``shared_weights`` and the weighted instructions.
    shared_weights : bool, optional
        Share one weight vector across edges. Defaults to ``True``.
    normalization : {"component", "integral", "norm"}, optional
        Spherical-harmonic normalization. Defaults to ``"component"``.

    Notes
    -----
    The second input is implicit: its direction defines the supplied frame.
    At the positive y-axis, only its order-zero component survives. Its
    constant value is folded into the CG coefficients during construction.
    Only nonzero couplings with ``m_in = +/- m_out`` are stored. The forward
    pass uses indexed products and sparse summation, without evaluating or
    rotating spherical harmonics or contracting a dense CG tensor.
    Rotation degrees follow the feature and output irreps. Shared Wigner
    matrices may cover additional degrees, but must retain all orders needed
    by those representations.
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Sequence[tuple],
        in1_var: Optional[Sequence[float]] = None,
        in2_var: Optional[Sequence[float]] = None,
        out_var: Optional[Sequence[float]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        *,
        normalization: str = "component",
    ) -> None:
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        if any(
            entry.mul != 1
            or entry.ir.p != (-1) ** entry.ir.l
            or getattr(entry.ir, "t", 1) != 1
            for entry in self.irreps_in2
        ):
            raise ValueError(
                "irreps_in2 must describe time-even edge spherical harmonics."
            )
        if any(ins[3] not in ("uvu", "uvw") for ins in instructions):
            raise ValueError("O3TensorProduct supports uvu and uvw paths.")
        if normalization not in ("component", "integral", "norm"):
            raise ValueError("normalization must be component, integral, or norm.")
        self.normalization = normalization

        # Resolve path conventions without building an executable CGTP.
        metadata = o3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            compile_left_right=False,
            compile_right=False,
        )
        self.instructions = metadata.instructions
        self.weight_numel = metadata.weight_numel
        self.weight_shape = (self.weight_numel,)
        self.internal_weights = metadata.internal_weights
        self.shared_weights = metadata.shared_weights
        if self.internal_weights and self.weight_numel:
            self.weight = metadata.weight
        else:
            self.register_buffer("weight", metadata.weight)
        self.register_buffer("output_mask", metadata.output_mask)

        self.local_frame_in = LocalFrame(self.irreps_in1)
        output_frame = LocalFrame(self.irreps_out, reverse=True)
        simplified_irreps_out = self.irreps_out.simplify()
        self.local_frame_out = LocalFrame(simplified_irreps_out, reverse=True)
        self.lmax = max(self.local_frame_in.lmax, self.local_frame_out.lmax)
        # Adjacent equal irreps share a single rotation over their channels.
        # The public output still preserves the declared, unsimplified layout.
        output_index = []
        simplified_offset = 0
        entry_index = 0
        for entry in simplified_irreps_out:
            ir, mul = entry.ir, entry.mul
            channel_offset = 0
            while channel_offset < mul:
                width = self.irreps_out[entry_index].mul
                output_index.extend(
                    simplified_offset + m * mul + channel_offset + channel
                    for m in range(ir.dim)
                    for channel in range(width)
                )
                channel_offset += width
                entry_index += 1
            simplified_offset += ir.dim * mul
        self._simplify_output = self.irreps_out != simplified_irreps_out
        self.register_buffer(
            "output_index",
            torch.tensor(output_index, dtype=torch.long),
            persistent=False,
        )
        self.input_dim = self.irreps_in1.dim
        self.local_output_dim = self.local_frame_out.output_dim
        self.num_harmonics = len(self.irreps_in2)
        self._has_unweighted = any(not ins.has_weight for ins in self.instructions)

        input_components = tuple(
            _entry_components(self.local_frame_in, i)
            for i in range(len(self.irreps_in1))
        )
        output_components = tuple(
            _entry_components(output_frame, i) for i in range(len(self.irreps_out))
        )
        contractions = {}
        offset = 0
        for ins in self.instructions:
            ir1, mul1 = self.irreps_in1[ins.i_in1].ir, self.irreps_in1[ins.i_in1].mul
            ir2 = self.irreps_in2[ins.i_in2].ir
            ir_out, mul_out = (
                self.irreps_out[ins.i_out].ir,
                self.irreps_out[ins.i_out].mul,
            )
            width = math.prod(ins.path_shape)
            weight_indices = (
                list(range(offset, offset + width))
                if ins.has_weight
                else [self.weight_numel] * width
            )
            offset += width if ins.has_weight else 0
            if not mul1 or not mul_out or not ins.path_weight:
                continue
            key = (ins.connection_mode, mul1, mul_out)
            contraction = contractions.setdefault(
                key, dict(input=[], output=[], weight=[], scale=[], harmonic=[])
            )
            cg = o3.wigner_3j(ir1.l, ir2.l, ir_out.l, dtype=torch.float64)[:, ir2.l, :]
            pole = math.sqrt(2 * ir2.l + 1) if normalization != "norm" else 1.0
            if normalization == "integral":
                pole /= math.sqrt(4 * math.pi)
            # In the real basis, only equal absolute magnetic orders can couple.
            for m_out in range(-min(ir1.l, ir_out.l), min(ir1.l, ir_out.l) + 1):
                for m_in in sorted({m_out, -m_out}):
                    coefficient = float(cg[ir1.l + m_in, ir_out.l + m_out])
                    if coefficient == 0.0:
                        continue
                    input_start, input_sign = input_components[ins.i_in1][m_in]
                    output_start, output_sign = output_components[ins.i_out][m_out]
                    contraction["input"].extend(range(input_start, input_start + mul1))
                    contraction["output"].extend(
                        range(output_start, output_start + mul_out)
                    )
                    contraction["weight"].extend(weight_indices)
                    contraction["scale"].append(
                        coefficient * pole * ins.path_weight * input_sign * output_sign
                    )
                    contraction["harmonic"].append(ins.i_in2)

        self._contractions = tuple(contractions)
        for index, contraction in enumerate(contractions.values()):
            for name, values in contraction.items():
                self.register_buffer(
                    f"{name}_{index}",
                    torch.tensor(
                        values,
                        dtype=torch.get_default_dtype()
                        if name == "scale"
                        else torch.long,
                    ),
                    persistent=False,
                )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.irreps_in1} x Y({self.irreps_in2}) "
            f"-> {self.irreps_out} | {self.weight_numel} weights)"
        )

    def forward_local(
        self,
        features: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        harmonic_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the tensor product in the local frame.

        Parameters
        ----------
        features : torch.Tensor
            Local features with shape ``(..., local_frame_in.irreps_out.dim)``.
        weight : torch.Tensor, optional
            External weights with shape ``(..., weight_numel)``. Leading
            dimensions broadcast with the features.
        harmonic_scale : torch.Tensor, optional
            Invariant amplitude for each spherical-harmonic entry, with shape
            ``(..., len(irreps_in2))``. Defaults to one.

        Returns
        -------
        torch.Tensor
            Features with shape ``(..., local_frame_out.irreps_out.dim)``.
        """
        if features.ndim < 1 or features.shape[-1] != self.local_frame_in.output_dim:
            raise ValueError(
                f"Local feature trailing dimension must be {self.local_frame_in.output_dim}."
            )
        if weight is None:
            if not self.internal_weights and self.weight_numel:
                raise RuntimeError(
                    "Weights must be provided when internal_weights=False."
                )
            weight = self.weight
        if weight.ndim < 1 or weight.shape[-1] != self.weight_numel:
            raise ValueError(f"Expected {self.weight_numel} weights.")
        if harmonic_scale is not None and (
            harmonic_scale.ndim < 1 or harmonic_scale.shape[-1] != self.num_harmonics
        ):
            raise ValueError(
                "harmonic_scale must contain one value per harmonic entry."
            )
        leading_shape = torch.broadcast_shapes(
            features.shape[:-1],
            weight.shape[:-1],
            () if harmonic_scale is None else harmonic_scale.shape[:-1],
        )
        if self._has_unweighted:
            weight = torch.cat(
                (weight, weight.new_ones((*weight.shape[:-1], 1))), dim=-1
            )
        output = None
        for index, (mode, mul1, mul_out) in enumerate(self._contractions):
            scale = getattr(self, f"scale_{index}")
            count = scale.numel()
            values = features.index_select(-1, getattr(self, f"input_{index}"))
            values = values.reshape(*features.shape[:-1], count, mul1)
            weights = weight.index_select(-1, getattr(self, f"weight_{index}"))
            if mode == "uvu":
                values = values * weights.reshape(*weight.shape[:-1], count, mul1)
            else:
                weights = weights.reshape(*weight.shape[:-1], count, mul1, mul_out)
                values = torch.matmul(values.unsqueeze(-2), weights).squeeze(-2)
            if harmonic_scale is not None:
                scale = scale * harmonic_scale.index_select(
                    -1, getattr(self, f"harmonic_{index}")
                )
            values = (values * scale.unsqueeze(-1)).flatten(-2)
            contribution = scatter(
                values,
                getattr(self, f"output_{index}"),
                dim=-1,
                dim_size=self.local_output_dim,
                reduce="sum",
            )
            output = contribution if output is None else output + contribution
        if output is None:
            zero = features[..., :0].sum() + weight[..., :0].sum()
            if harmonic_scale is not None:
                zero = zero + harmonic_scale[..., :0].sum()
            return features.new_zeros((*leading_shape, self.local_output_dim)) + zero
        return output.expand(*leading_shape, self.local_output_dim)

    def forward(
        self,
        features: torch.Tensor,
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        harmonic_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rotate features, apply the tensor product, and rotate back.

        Parameters
        ----------
        features : torch.Tensor
            Global features with shape ``(batch, ..., irreps_in1.dim)``.
        wigner : torch.Tensor
            Global-to-local matrices with shape ``(batch, local_dim, global_dim)``.
            They must align the harmonic direction to the positive y-axis and
            retain all orders of the input and output representations.
        wigner_inv : torch.Tensor
            Local-to-global matrices with shape ``(batch, global_dim, local_dim)``.
        weight : torch.Tensor, optional
            External weights with shape ``(..., weight_numel)`` in instruction
            order. Leading dimensions broadcast with the features.
        harmonic_scale : torch.Tensor, optional
            Invariant amplitudes with shape ``(..., len(irreps_in2))``.

        Returns
        -------
        torch.Tensor
            Global output with shape ``(batch, ..., irreps_out.dim)``.
        """
        features = self.local_frame_in.to_local(features, wigner)
        features = self.forward_local(features, weight, harmonic_scale)
        features = self.local_frame_out.to_global(features, wigner_inv)
        if self._simplify_output:
            features = features.index_select(-1, self.output_index)
        return features
