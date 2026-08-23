################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from collections import Counter
from typing import Optional, Sequence, Tuple

import torch

from .irreps import Irrep, Irreps, IrrepsLike
from .projector import project as project_cartesian
from .utils import levi_civita


def _cartesian_product(
    input1: torch.Tensor,
    irrep1: Irrep,
    input2: torch.Tensor,
    irrep2: Irrep,
    irrep_out: Irrep,
    levi_civita: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one normalized delta or Levi-Civita Cartesian path."""
    difference = irrep1.l + irrep2.l - irrep_out.l
    if difference < 0 or irrep_out not in irrep1 * irrep2:
        raise ValueError(
            f"Illegal Cartesian O(3) path: {irrep1} x {irrep2} -> {irrep_out}."
        )

    if difference % 2 == 0:
        contracted = difference // 2
        free1 = 3 ** (irrep1.l - contracted)
        shared = 3**contracted
        free2 = 3 ** (irrep2.l - contracted)
        first = input1.reshape(input1.shape[:-2] + (free1, shared, input1.shape[-1]))
        second = input2.reshape(
            input2.shape[:-2] + (shared, free2, input2.shape[-1])
        )
        output = torch.einsum("...iac,...ajd->...ijcd", first, second)
        return output.reshape(
            output.shape[:-4]
            + (irrep_out.dim, input1.shape[-1], input2.shape[-1])
        ) * (3.0 ** (-0.5 * contracted))

    contracted = (difference - 1) // 2
    free1 = 3 ** (irrep1.l - contracted - 1)
    shared = 3**contracted
    free2 = 3 ** (irrep2.l - contracted - 1)
    first = input1.reshape(
        input1.shape[:-2] + (free1, 3, shared, input1.shape[-1])
    )
    second = input2.reshape(
        input2.shape[:-2] + (shared, 3, free2, input2.shape[-1])
    )
    output = torch.einsum(
        "...iuac,wuv,...avjd->...iwjcd",
        first,
        levi_civita,
        second,
    )
    return output.reshape(
        output.shape[:-5]
        + (irrep_out.dim, input1.shape[-1], input2.shape[-1])
    ) * (0.5 * 3.0**contracted) ** -0.5


class TensorProduct(torch.nn.Module):
    """A weighted complete O(3) Cartesian tensor product.

    Args:
        irreps_in1: First input irreps.
        irreps_in2: Second input irreps.
        irreps_out: Output irreps.
        channels_in1: First input channel count.
        channels_in2: Second input channel count.
        channels_out: Output channel count.
        project: This argument is mandatory. If ``True``, every output is
            projected onto its symmetric traceless irreducible subspace. If
            ``False``, the unprojected Cartesian result is returned so that a
            linear aggregation can precede :meth:`project_output`.
        path_mode: One of ``"u1u"``, ``"uuu"``, or ``"uvw"``.

    Inputs and outputs use ``(..., irreps.dim, channels)``. Delta contractions
    generate the even rank-difference paths and Levi-Civita contractions
    generate the odd rank-difference paths required by complete O(3).
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
        project: bool,
        path_mode: str = "uuu",
        internal_weights: bool = True,
        path_norm: bool = True,
        path: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(project, bool):
            raise TypeError("project must be explicitly set to True or False.")
        self.project = project
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        if channels_in2 is None:
            channels_in2 = 1 if path_mode == "u1u" else channels_in1
        if channels_out is None:
            channels_out = channels_in1
        channels = (channels_in1, channels_in2, channels_out)
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in channels
        ):
            raise TypeError("TensorProduct channel counts must be integers.")
        if any(value < 1 for value in channels):
            raise ValueError("TensorProduct channel counts must be positive.")
        if path_mode not in ("u1u", "uuu", "uvw"):
            raise ValueError("path_mode must be 'u1u', 'uuu', or 'uvw'.")
        if path_mode == "u1u" and (channels_in2 != 1 or channels_in1 != channels_out):
            raise ValueError(
                "path_mode='u1u' requires channels_in2=1 and equal first/output "
                "channel counts."
            )
        if path_mode == "uuu" and len(set(channels)) != 1:
            raise ValueError("path_mode='uuu' requires all channel counts to match.")
        self.channels_in1 = channels_in1
        self.channels_in2 = channels_in2
        self.channels_out = channels_out
        self.path_mode = path_mode
        self.internal_weights = bool(internal_weights)
        self.path_norm = bool(path_norm)
        self.register_buffer(
            "levi_civita",
            levi_civita(),
            persistent=False,
        )

        inputs1 = self.irreps_in1.expanded()
        inputs2 = self.irreps_in2.expanded()
        outputs = self.irreps_out.expanded()
        if path is None:
            paths = tuple(
                (output_index, input1_index, input2_index)
                for output_index, output in enumerate(outputs)
                for input1_index, input1_irrep in enumerate(inputs1)
                for input2_index, input2_irrep in enumerate(inputs2)
                if output in input1_irrep * input2_irrep
            )
        else:
            paths = tuple(path)
        for item in paths:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError("TensorProduct paths must contain three indices.")
            output_index, input1_index, input2_index = item
            if not 0 <= output_index < len(outputs):
                raise ValueError(f"Invalid TensorProduct output index: {output_index}.")
            if not 0 <= input1_index < len(inputs1):
                raise ValueError(f"Invalid TensorProduct input1 index: {input1_index}.")
            if not 0 <= input2_index < len(inputs2):
                raise ValueError(f"Invalid TensorProduct input2 index: {input2_index}.")
            if outputs[output_index] not in (
                inputs1[input1_index] * inputs2[input2_index]
            ):
                raise ValueError("Illegal complete O(3) TensorProduct path.")
        self.path = paths
        self._input1_slices = self.irreps_in1.expanded_slices()
        self._input2_slices = self.irreps_in2.expanded_slices()
        paths_by_output = [[] for _ in outputs]
        for path_index, (output_index, input1_index, input2_index) in enumerate(paths):
            paths_by_output[output_index].append(
                (path_index, input1_index, input2_index)
            )
        self._paths_by_output = tuple(tuple(group) for group in paths_by_output)

        counts = Counter(output_index for output_index, _, _ in paths)
        self.register_buffer(
            "path_scales",
            torch.tensor(
                [
                    counts[output_index] ** -0.5 if path_norm else 1.0
                    for output_index, _, _ in paths
                ],
                dtype=torch.get_default_dtype(),
            ),
            persistent=False,
        )
        if path_mode in ("u1u", "uuu"):
            self.weight_shape = (len(paths), channels_out)
        else:
            self.weight_shape = (
                len(paths),
                channels_in1,
                channels_in2,
                channels_out,
            )
        self.weight_numel = math.prod(self.weight_shape)
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.empty(self.weight_shape))
            fan_in = (
                1
                if path_mode in ("u1u", "uuu")
                else channels_in1 * channels_in2
            )
            torch.nn.init.uniform_(
                self.weight, -math.sqrt(3.0 / fan_in), math.sqrt(3.0 / fan_in)
            )
        else:
            self.register_parameter("weight", None)

    def _resolve_weight(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if self.internal_weights:
            if weight is not None:
                raise ValueError("Do not pass weight when using internal weights.")
            weight = self.weight
        elif weight is None:
            raise ValueError("TensorProduct requires external weight.")
        if weight is None:
            raise RuntimeError("TensorProduct weight resolution failed.")
        if weight.is_complex():
            raise TypeError("Cartesian O(3) TensorProduct supports real weights only.")
        ndim = len(self.weight_shape)
        if weight.ndim < ndim or tuple(weight.shape[-ndim:]) != self.weight_shape:
            raise ValueError(
                "TensorProduct weight trailing shape must be "
                f"{self.weight_shape}, got {tuple(weight.shape)}."
            )
        return weight

    def project_output(self, input: torch.Tensor) -> torch.Tensor:
        """Project a dense unprojected output onto every output irrep."""
        expected = (self.irreps_out.dim, self.channels_out)
        if input.ndim < 2 or tuple(input.shape[-2:]) != expected:
            raise ValueError(
                f"Projection input trailing shape must be {expected}, "
                f"got {tuple(input.shape)}."
            )
        if not len(self.irreps_out):
            return input
        return torch.cat(
            [
                project_cartesian(input[..., block_slice, :], irrep.l)
                for irrep, block_slice in zip(
                    self.irreps_out.expanded(), self.irreps_out.expanded_slices()
                )
            ],
            dim=-2,
        )

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expected1 = (self.irreps_in1.dim, self.channels_in1)
        expected2 = (self.irreps_in2.dim, self.channels_in2)
        if input1.is_complex() or input2.is_complex():
            raise TypeError("Cartesian O(3) TensorProduct supports real inputs only.")
        if input1.ndim < 2 or tuple(input1.shape[-2:]) != expected1:
            raise ValueError(f"input1 trailing shape must be {expected1}.")
        if input2.ndim < 2 or tuple(input2.shape[-2:]) != expected2:
            raise ValueError(f"input2 trailing shape must be {expected2}.")
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
                "TensorProduct inputs and weight batches do not broadcast."
            ) from error
        input1 = input1.expand(leading_shape + expected1)
        input2 = input2.expand(leading_shape + expected2)
        weight = weight.expand(leading_shape + self.weight_shape)
        inputs1 = self.irreps_in1.expanded()
        inputs2 = self.irreps_in2.expanded()
        outputs = self.irreps_out.expanded()
        zero = input1.sum() * 0 + input2.sum() * 0 + weight.sum() * 0
        blocks = []
        for output_index, output_irrep in enumerate(outputs):
            contributions = []
            for path_index, input1_index, input2_index in self._paths_by_output[
                output_index
            ]:
                product = _cartesian_product(
                    input1[..., self._input1_slices[input1_index], :],
                    inputs1[input1_index],
                    input2[..., self._input2_slices[input2_index], :],
                    inputs2[input2_index],
                    output_irrep,
                    self.levi_civita,
                )
                if self.path_mode == "uuu":
                    value = torch.diagonal(product, dim1=-2, dim2=-1)
                    value = value * weight[..., path_index, :].unsqueeze(-2)
                elif self.path_mode == "u1u":
                    value = product.squeeze(-1)
                    value = value * weight[..., path_index, :].unsqueeze(-2)
                else:
                    value = torch.einsum(
                        "...duv,...uvc->...dc",
                        product,
                        weight[..., path_index, :, :, :],
                    )
                contributions.append(value * self.path_scales[path_index])
            if contributions:
                block = sum(contributions[1:], contributions[0])
            else:
                block = input1.new_zeros(
                    leading_shape + (output_irrep.dim, self.channels_out)
                )
                block = block + zero
            blocks.append(block)
        if blocks:
            output = torch.cat(blocks, dim=-2)
        else:
            output = input1.new_empty(leading_shape + (0, self.channels_out)) + zero
        return self.project_output(output) if self.project else output

    def extra_repr(self) -> str:
        return (
            f"irreps_in1={self.irreps_in1}, irreps_in2={self.irreps_in2}, "
            f"irreps_out={self.irreps_out}, channels_in1={self.channels_in1}, "
            f"channels_in2={self.channels_in2}, channels_out={self.channels_out}, "
            f"project={self.project}, path_mode={self.path_mode}, "
            f"num_paths={len(self.path)}"
        )
