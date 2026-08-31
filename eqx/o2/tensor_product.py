################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from collections import Counter
from typing import Optional, Sequence, Tuple

import torch

from .irreps import Irrep, Irreps, IrrepsLike


def _quarter_turn(input: torch.Tensor) -> torch.Tensor:
    return torch.stack((-input[..., 1, :], input[..., 0, :]), dim=-2)


def _cg_product(
    input1: torch.Tensor,
    irrep1: Irrep,
    input2: torch.Tensor,
    irrep2: Irrep,
    irrep_out: Irrep,
) -> torch.Tensor:
    """Evaluate one orthonormal real O(2) Clebsch--Gordan map."""
    if irrep_out not in irrep1 * irrep2:
        raise ValueError(
            f"Illegal O(2) tensor-product path: {irrep1} x {irrep2} -> {irrep_out}."
        )

    if irrep1.m == 0 and irrep2.m == 0:
        value = input1[..., 0, :, None] * input2[..., 0, None, :]
        return value.unsqueeze(-3)

    if irrep1.m == 0:
        vector = input2
        if irrep1.p == -1:
            vector = _quarter_turn(vector)
        return input1[..., 0, None, :, None] * vector.unsqueeze(-2)

    if irrep2.m == 0:
        vector = input1
        if irrep2.p == -1:
            vector = _quarter_turn(vector)
        return vector.unsqueeze(-1) * input2[..., 0, None, None, :]

    first_real = input1[..., 0, :, None]
    first_imag = input1[..., 1, :, None]
    second_real = input2[..., 0, None, :]
    second_imag = input2[..., 1, None, :]
    scale = math.sqrt(0.5)

    if irrep_out.m == irrep1.m + irrep2.m:
        real = first_real * second_real - first_imag * second_imag
        imag = first_real * second_imag + first_imag * second_real
        return torch.stack((real, imag), dim=-3) * scale

    difference_real = first_real * second_real + first_imag * second_imag
    difference_imag = first_imag * second_real - first_real * second_imag
    if irrep2.m > irrep1.m:
        difference_imag = -difference_imag
    if irrep_out.m > 0:
        return torch.stack((difference_real, difference_imag), dim=-3) * scale
    if irrep_out.is_even_scalar():
        return difference_real.unsqueeze(-3) * scale
    return difference_imag.unsqueeze(-3) * scale


def _cg_product_uuu(
    input1: torch.Tensor,
    irrep1: Irrep,
    input2: torch.Tensor,
    irrep2: Irrep,
    irrep_out: Irrep,
) -> torch.Tensor:
    """Evaluate a channel-wise real O(2) Clebsch--Gordan map."""
    if irrep_out not in irrep1 * irrep2:
        raise ValueError(
            f"Illegal O(2) tensor-product path: {irrep1} x {irrep2} -> {irrep_out}."
        )
    if input1.size(-1) != input2.size(-1):
        raise ValueError("uuu tensor products require equal channel counts.")

    if irrep1.m == 0 and irrep2.m == 0:
        return (input1[..., 0, :] * input2[..., 0, :]).unsqueeze(-2)

    if irrep1.m == 0:
        vector = input2
        if irrep1.p == -1:
            vector = _quarter_turn(vector)
        return input1[..., 0, :].unsqueeze(-2) * vector

    if irrep2.m == 0:
        vector = input1
        if irrep2.p == -1:
            vector = _quarter_turn(vector)
        return vector * input2[..., 0, :].unsqueeze(-2)

    first_real = input1[..., 0, :]
    first_imag = input1[..., 1, :]
    second_real = input2[..., 0, :]
    second_imag = input2[..., 1, :]
    scale = math.sqrt(0.5)

    if irrep_out.m == irrep1.m + irrep2.m:
        real = first_real * second_real - first_imag * second_imag
        imag = first_real * second_imag + first_imag * second_real
        return torch.stack((real, imag), dim=-2) * scale

    difference_real = first_real * second_real + first_imag * second_imag
    difference_imag = first_imag * second_real - first_real * second_imag
    if irrep2.m > irrep1.m:
        difference_imag = -difference_imag
    if irrep_out.m > 0:
        return torch.stack((difference_real, difference_imag), dim=-2) * scale
    if irrep_out.is_even_scalar():
        return difference_real.unsqueeze(-2) * scale
    return difference_imag.unsqueeze(-2) * scale


class TensorProduct(torch.nn.Module):
    """A weighted tensor product between real O(2) representations.

    Inputs use shapes ``(..., irreps_in1.dim, channels_in1)`` and
    ``(..., irreps_in2.dim, channels_in2)``. ``path_mode="u1u"`` requires
    one channel in the second input and preserves the first input channels.
    ``path_mode="uuu"`` contracts equal channel indices and requires all
    channel counts to match.
    ``path_mode="uvw"`` learns a dense channel tensor for every equivariant
    path. Paths are ``(output_index, input1_index, input2_index)`` tuples over
    expanded irrep-copy layouts.
    """

    def __init__(
        self,
        irreps_in1: IrrepsLike,
        irreps_in2: IrrepsLike,
        irreps_out: IrrepsLike,
        channels_in1: int,
        channels_in2: Optional[int] = None,
        channels_out: Optional[int] = None,
        *,
        path_mode: str = "uuu",
        internal_weights: bool = True,
        path_norm: bool = True,
        path: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> None:
        super().__init__()

        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        if channels_in2 is None:
            channels_in2 = 1 if path_mode == "u1u" else channels_in1
        if channels_out is None:
            channels_out = channels_in1
        channel_counts = (channels_in1, channels_in2, channels_out)
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in channel_counts
        ):
            raise TypeError("TensorProduct channel counts must be integers.")
        if any(value < 1 for value in channel_counts):
            raise ValueError("TensorProduct channel counts must be positive.")
        if path_mode not in ("u1u", "uuu", "uvw"):
            raise ValueError("path_mode must be 'u1u', 'uuu', or 'uvw'.")
        if path_mode == "u1u" and (channels_in2 != 1 or channels_in1 != channels_out):
            raise ValueError(
                "path_mode='u1u' requires channels_in2 == 1 and "
                "channels_in1 == channels_out."
            )
        if path_mode == "uuu" and len(set(channel_counts)) != 1:
            raise ValueError("path_mode='uuu' requires all channel counts to match.")

        self.channels_in1 = channels_in1
        self.channels_in2 = channels_in2
        self.channels_out = channels_out
        self.path_mode = path_mode
        self.internal_weights = bool(internal_weights)
        self.path_norm = bool(path_norm)

        input1_irrep_list = self.irreps_in1.expanded()
        input2_irrep_list = self.irreps_in2.expanded()
        output_irrep_list = self.irreps_out.expanded()
        if path is None:
            paths = tuple(
                (output_index, input1_index, input2_index)
                for output_index, output_ir in enumerate(output_irrep_list)
                for input1_index, input1_ir in enumerate(input1_irrep_list)
                for input2_index, input2_ir in enumerate(input2_irrep_list)
                if output_ir in input1_ir * input2_ir
            )
        else:
            paths = tuple(path)
        for item in paths:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError(
                    "Each TensorProduct path must be "
                    "(output_index, input1_index, input2_index)."
                )
            output_index, input1_index, input2_index = item
            if any(
                not isinstance(index, int) or isinstance(index, bool) for index in item
            ):
                raise TypeError("TensorProduct path indices must be integers.")
            if not 0 <= output_index < len(output_irrep_list):
                raise ValueError(f"Invalid TensorProduct output index: {output_index}.")
            if not 0 <= input1_index < len(input1_irrep_list):
                raise ValueError(f"Invalid TensorProduct input1 index: {input1_index}.")
            if not 0 <= input2_index < len(input2_irrep_list):
                raise ValueError(f"Invalid TensorProduct input2 index: {input2_index}.")
            if output_irrep_list[output_index] not in (
                input1_irrep_list[input1_index] * input2_irrep_list[input2_index]
            ):
                raise ValueError(
                    "Illegal O(2) TensorProduct path: "
                    f"{input1_irrep_list[input1_index]} x "
                    f"{input2_irrep_list[input2_index]} "
                    f"-> {output_irrep_list[output_index]}."
                )
        self.path = paths

        path_counts = Counter(output_index for output_index, _, _ in paths)
        scales = [
            path_counts[output_index] ** -0.5 if self.path_norm else 1.0
            for output_index, _, _ in paths
        ]
        self.register_buffer(
            "path_scales",
            torch.tensor(scales, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        paths_by_output = [[] for _ in output_irrep_list]
        for path_index, (output_index, input1_index, input2_index) in enumerate(paths):
            paths_by_output[output_index].append(
                (path_index, input1_index, input2_index)
            )
        self._paths_by_output = tuple(tuple(group) for group in paths_by_output)
        self._input1_slices = self.irreps_in1.expanded_slices()
        self._input2_slices = self.irreps_in2.expanded_slices()

        if self.path_mode in ("u1u", "uuu"):
            self.weight_shape = (len(paths), self.channels_out)
        else:
            self.weight_shape = (
                len(paths),
                self.channels_in1,
                self.channels_in2,
                self.channels_out,
            )
        self.weight_numel = math.prod(self.weight_shape)
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.empty(self.weight_shape))
            if self.weight_numel:
                fan_in = (
                    1
                    if self.path_mode in ("u1u", "uuu")
                    else self.channels_in1 * self.channels_in2
                )
                bound = math.sqrt(3.0 / fan_in)
                torch.nn.init.uniform_(self.weight, -bound, bound)
        else:
            self.register_parameter("weight", None)

    def _resolve_weight(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if self.internal_weights:
            if weight is not None:
                raise ValueError(
                    "Do not pass weight when TensorProduct uses internal weights."
                )
            weight = self.weight
        elif weight is None:
            raise ValueError("TensorProduct requires external weight.")
        if weight is None:
            raise RuntimeError("TensorProduct weight resolution failed.")
        if weight.is_complex():
            raise TypeError("O(2) TensorProduct supports real weights only.")
        weight_ndim = len(self.weight_shape)
        if (
            weight.ndim < weight_ndim
            or tuple(weight.shape[-weight_ndim:]) != self.weight_shape
        ):
            raise ValueError(
                "TensorProduct weight trailing shape must be "
                f"{self.weight_shape}, got {tuple(weight.shape)}."
            )
        return weight

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input1.is_complex() or input2.is_complex():
            raise TypeError("O(2) TensorProduct supports real inputs only.")
        expected1 = (self.irreps_in1.dim, self.channels_in1)
        expected2 = (self.irreps_in2.dim, self.channels_in2)
        if input1.ndim < 2 or tuple(input1.shape[-2:]) != expected1:
            raise ValueError(
                f"TensorProduct input1 trailing shape must be {expected1}, "
                f"got {tuple(input1.shape)}."
            )
        if input2.ndim < 2 or tuple(input2.shape[-2:]) != expected2:
            raise ValueError(
                f"TensorProduct input2 trailing shape must be {expected2}, "
                f"got {tuple(input2.shape)}."
            )
        weight = self._resolve_weight(weight)
        weight_ndim = len(self.weight_shape)
        try:
            leading_shape = torch.broadcast_shapes(
                input1.shape[:-2],
                input2.shape[:-2],
                weight.shape[:-weight_ndim],
            )
        except RuntimeError as error:
            raise ValueError(
                "TensorProduct input and weight batch dimensions do not broadcast."
            ) from error
        input1 = input1.expand(leading_shape + expected1)
        input2 = input2.expand(leading_shape + expected2)

        input1_irrep_list = self.irreps_in1.expanded()
        input2_irrep_list = self.irreps_in2.expanded()
        output_irrep_list = self.irreps_out.expanded()
        zero_dependency = input1.sum() * 0 + input2.sum() * 0 + weight.sum() * 0
        output_blocks = []
        for output_index, output_ir in enumerate(output_irrep_list):
            contributions = []
            for path_index, input1_index, input2_index in self._paths_by_output[
                output_index
            ]:
                if self.path_mode == "uuu":
                    contribution = _cg_product_uuu(
                        input1[..., self._input1_slices[input1_index], :],
                        input1_irrep_list[input1_index],
                        input2[..., self._input2_slices[input2_index], :],
                        input2_irrep_list[input2_index],
                        output_ir,
                    )
                    contribution = contribution * weight[..., path_index, :].unsqueeze(
                        -2
                    )
                else:
                    product = _cg_product(
                        input1[..., self._input1_slices[input1_index], :],
                        input1_irrep_list[input1_index],
                        input2[..., self._input2_slices[input2_index], :],
                        input2_irrep_list[input2_index],
                        output_ir,
                    )
                    if self.path_mode == "u1u":
                        contribution = product.squeeze(-1)
                        contribution = contribution * weight[
                            ..., path_index, :
                        ].unsqueeze(-2)
                    else:
                        contribution = torch.einsum(
                            "...duv,...uvc->...dc",
                            product,
                            weight[..., path_index, :, :, :],
                        )
                contributions.append(contribution * self.path_scales[path_index])
            if contributions:
                output_block = sum(contributions[1:], contributions[0])
            else:
                output_block = input1.new_zeros(
                    leading_shape + (output_ir.dim, self.channels_out)
                )
                output_block = output_block + zero_dependency
            output_blocks.append(output_block)

        if output_blocks:
            return torch.cat(output_blocks, dim=-2)
        return (
            input1.new_empty(leading_shape + (0, self.channels_out)) + zero_dependency
        )

    def extra_repr(self) -> str:
        return (
            f"irreps_in1={self.irreps_in1}, irreps_in2={self.irreps_in2}, "
            f"irreps_out={self.irreps_out}, channels_in1={self.channels_in1}, "
            f"channels_in2={self.channels_in2}, channels_out={self.channels_out}, "
            f"path_mode={self.path_mode}, internal_weights={self.internal_weights}, "
            f"num_paths={len(self.path)}"
        )
