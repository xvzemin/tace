"""Channel-linear maps for Cartesian O(2) tensors."""

import math

import torch

from eqx import o2

from .basis import ChangeOfBasis, Projector
from .irreps import Irreps


class Linear(torch.nn.Module):
    """Mix channels carrying identical irreps.

    Parameters
    ----------
    irreps_in, irreps_out : Irreps or str
        Cartesian input and output representations.
    internal_weights, shared_weights : bool, optional
        Store parameters and share weights over leading dimensions.
    instructions : list of tuple of int, optional
        Input/output entry pairs. Defaults to all matching irreps.
    biases : bool or sequence of bool, optional
        Biases on reflection- and time-even scalar outputs.
    path_normalization : {"element", "path"}, optional
        Normalize by input channel count or incoming path count.
    project : bool, optional
        Project Cartesian outputs onto their STF subspaces.
    output_basis : {"cartesian", "circular"}, optional
        Cartesian mul_ir or compact circular ir_mul output storage.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        internal_weights=None,
        shared_weights=None,
        instructions=None,
        biases=False,
        path_normalization="element",
        project=False,
        output_basis="cartesian",
    ):
        super().__init__()
        if output_basis not in ("cartesian", "circular"):
            raise ValueError("output_basis must be 'cartesian' or 'circular'.")
        self.irreps_in = Irreps(irreps_in)
        self.cartesian_out = Irreps(irreps_out)
        self.irreps_out = (
            self.cartesian_out
            if output_basis == "cartesian"
            else self.cartesian_out.circular()
        )
        self.output_basis = output_basis
        self.input_dim = self.irreps_in.dim
        self.input_paths = tuple(
            (mul, ir.dim, section)
            for (mul, ir), section in zip(self.irreps_in, self.irreps_in.slices())
        )
        self.output_shapes = tuple((mul, ir.dim) for mul, ir in self.cartesian_out)
        reference = o2.Linear(
            self.irreps_in.circular(),
            self.cartesian_out.circular(),
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            instructions=instructions,
            biases=biases,
            path_normalization=path_normalization,
        )
        for name in (
            "instructions",
            "weight_numel",
            "weight_shape",
            "bias_numel",
            "internal_weights",
            "shared_weights",
            "path_normalization",
        ):
            setattr(self, name, getattr(reference, name))
        for name in ("weight", "bias"):
            value = getattr(reference, name)
            if isinstance(value, torch.nn.Parameter):
                self.register_parameter(name, value)
            else:
                self.register_buffer(name, value)
        if output_basis == "circular":
            self.projection = ChangeOfBasis(self.cartesian_out, inverse=True)
        elif project:
            self.projection = Projector(self.cartesian_out)
        else:
            self.projection = torch.nn.Identity()
        connected = {ins.i_out for ins in self.instructions if ins.path_weight}
        self.register_buffer(
            "output_mask",
            (
                torch.cat(
                    [
                        torch.full((entry.dim,), float(i in connected))
                        for i, entry in enumerate(self.irreps_out)
                    ]
                )
                if self.irreps_out
                else torch.empty(0)
            ),
            persistent=False,
        )

    def forward(self, features, weight=None, bias=None):
        """Apply the linear map with optional external parameters.

        Parameters
        ----------
        features : torch.Tensor
            Cartesian features of shape ``(..., irreps_in.dim)`` in ``mul_ir``
            layout.
        weight, bias : torch.Tensor, optional
            Parameters with trailing sizes ``weight_numel`` and ``bias_numel``.
            Leading dimensions broadcast with the input. Defaults to the
            internal parameters.

        Returns
        -------
        torch.Tensor
            Features of shape ``(..., irreps_out.dim)`` in the selected
            ``output_basis``.
        """
        if features.shape[-1] != self.input_dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        if weight.shape[-1] != self.weight_numel or bias.shape[-1] != self.bias_numel:
            raise ValueError("Weight or bias size does not match the instructions.")
        shape = torch.broadcast_shapes(
            features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
        )
        zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
        inputs = [
            features[..., section].reshape(*features.shape[:-1], mul, dim)
            for mul, dim, section in self.input_paths
        ]
        outputs = [
            features.new_zeros((*shape, mul, dim)) + zero
            for mul, dim in self.output_shapes
        ]
        offset, bias_offset = 0, 0
        for ins in self.instructions:
            size = math.prod(ins.path_shape)
            if ins.i_in == -1:
                value = bias[..., bias_offset : bias_offset + size].unsqueeze(-1)
                bias_offset += size
            else:
                w = weight[..., offset : offset + size].reshape(
                    *weight.shape[:-1], *ins.path_shape
                )
                offset += size
                value = torch.einsum("...ud,...uv->...vd", inputs[ins.i_in], w)
            outputs[ins.i_out] = outputs[ins.i_out] + value * ins.path_weight
        output = (
            torch.cat([v.flatten(-2) for v in outputs], -1)
            if outputs
            else features.new_zeros((*shape, 0)) + zero
        )
        return self.projection(output)

    def weight_view_for_instruction(self, instruction, weight=None):
        """Return an instruction-shaped view of external or internal weights."""
        ins = self.instructions[instruction]
        if ins.i_in == -1:
            raise ValueError("Bias instructions have no weights.")
        weight = self.weight if weight is None else weight
        offset = sum(
            math.prod(p.path_shape)
            for p in self.instructions[:instruction]
            if p.i_in != -1
        )
        return weight[..., offset : offset + math.prod(ins.path_shape)].reshape(
            *weight.shape[:-1], *ins.path_shape
        )

    def weight_views(self, weight=None, yield_instruction=False):
        """Iterate over non-bias instruction weights."""
        for i, ins in enumerate(self.instructions):
            if ins.i_in != -1:
                value = self.weight_view_for_instruction(i, weight)
                yield (i, ins, value) if yield_instruction else value

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights"


class UuLinear(torch.nn.Module):
    """Mix equivalent Cartesian O(2) copies channel-wise.

    Parameters
    ----------
    irreps_in, irreps_out : Irreps or str
        Representations in flattened ``mul_ir`` storage.
    num_channel : int
        Shared channel count, dividing each input and output multiplicity.
    path_mode : {"sum", "expand"}, optional
        Sum paths or retain each input/output-copy path separately.
    """

    def __init__(self, irreps_in, irreps_out, num_channel, *, path_mode="sum"):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        reference = o2.UuLinear(
            self.irreps_in.circular(),
            Irreps(irreps_out).circular(),
            num_channel,
            path_mode=path_mode,
        )
        self.irreps_out = Irreps(reference.irreps_out)
        self.num_channel, self.path_mode = num_channel, path_mode
        self.instructions = reference.instructions
        self.weight_numel = reference.weight_numel
        self.weight_shape = (self.weight_numel,)
        self.slices = self.irreps_in.slices()

    def forward(self, features, weight):
        """Apply weights ordered as (copies_in, copies_out, num_channel)."""
        if (
            features.shape[-1] != self.irreps_in.dim
            or weight.shape[-1] != self.weight_numel
        ):
            raise ValueError("Incorrect feature or weight dimension.")
        shape = torch.broadcast_shapes(features.shape[:-1], weight.shape[:-1])
        outputs = [[] for _ in self.irreps_out]
        offset = 0
        for ins in self.instructions:
            _, ir = self.irreps_in[ins.i_in]
            ni, no, channels = ins.path_shape
            x = features[..., self.slices[ins.i_in]].reshape(
                *features.shape[:-1], ni, channels, ir.dim
            )
            size = ni * no * channels
            w = weight[..., offset : offset + size].reshape(
                *weight.shape[:-1], ni, no, channels
            )
            offset += size
            value = (
                torch.einsum(
                    "...iud,...iou->...ioud"
                    if self.path_mode == "expand"
                    else "...iud,...iou->...oud",
                    x,
                    w,
                )
                * ins.path_weight
            )
            outputs[ins.i_out].append(
                value.flatten(-4, -2)
                if self.path_mode == "expand"
                else value.flatten(-3, -2)
            )
        zero = features[..., :0].sum() + weight[..., :0].sum()
        values = []
        for (mul, ir), parts in zip(self.irreps_out, outputs):
            value = (
                (
                    torch.cat(parts, -2)
                    if self.path_mode == "expand"
                    else sum(parts[1:], parts[0])
                )
                if parts
                else features.new_zeros((*shape, mul, ir.dim)) + zero
            )
            values.append(value.flatten(-2))
        return (
            torch.cat(values, -1) if values else features.new_zeros((*shape, 0)) + zero
        )
