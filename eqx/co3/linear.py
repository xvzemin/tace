"""Channel-linear maps in Cartesian tensor storage."""

import math
from typing import NamedTuple

import torch

from .basis import ChangeOfBasis, Projector
from .irreps import Irreps


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


class Linear(torch.nn.Module):
    """Mix channels of identical irreps.

    Parameters
    ----------
    irreps_in, irreps_out : Irreps or str
        Input and output representations. Entry partitions are retained.
    internal_weights : bool, optional
        Store trainable weights. Defaults to True unless shared_weights=False.
    shared_weights : bool, optional
        Share weights across leading feature dimensions. Defaults to True.
    instructions : sequence of (int, int), optional
        Input/output entry pairs. Defaults to all equal-irrep pairs.
    biases : bool or sequence of bool, optional
        Add biases to invariant scalar outputs.
    path_normalization : {"element", "path"}, optional
        Normalize by input channel count or by incoming path count.
    project : bool, optional
        Project outputs onto their symmetric traceless subspaces.
    output_basis : {"cartesian", "spherical"}, optional
        Output storage. Spherical output applies the transposed path matrix
        after channel mixing, including projection of raw Cartesian inputs.
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
        if path_normalization not in ("element", "path"):
            raise ValueError("path_normalization must be 'element' or 'path'.")
        if output_basis not in ("cartesian", "spherical"):
            raise ValueError("output_basis must be 'cartesian' or 'spherical'.")
        self.irreps_in = Irreps(irreps_in)
        cartesian_out = Irreps(irreps_out)
        self.irreps_out = (
            cartesian_out.spherical() if output_basis == "spherical" else cartesian_out
        )
        self.output_basis = output_basis
        self.path_normalization = path_normalization
        self.shared_weights = True if shared_weights is None else shared_weights
        self.internal_weights = (
            self.shared_weights if internal_weights is None else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("Internal weights require shared_weights=True.")
        if instructions is None:
            instructions = [
                (i, j)
                for i, (_, a) in enumerate(self.irreps_in)
                for j, (_, b) in enumerate(cartesian_out)
                if a == b
            ]
        instructions = list(instructions)
        paths = []
        for i, j in instructions:
            if not 0 <= i < len(self.irreps_in) or not 0 <= j < len(cartesian_out):
                raise IndexError("Linear instruction index is out of range.")
            if self.irreps_in[i].ir != cartesian_out[j].ir:
                raise ValueError("Linear paths must connect identical irreps.")
            denominator = sum(
                self.irreps_in[k if path_normalization == "element" else i].mul
                for k, dest in instructions
                if dest == j
            )
            paths.append(
                Instruction(
                    i,
                    j,
                    (self.irreps_in[i].mul, cartesian_out[j].mul),
                    max(denominator, 1) ** -0.5,
                )
            )
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in cartesian_out]
        if len(biases) != len(cartesian_out) or any(
            b and not ir.is_scalar() for b, (_, ir) in zip(biases, cartesian_out)
        ):
            raise ValueError("Biases must select only invariant scalar outputs.")
        self.instructions = paths + [
            Instruction(-1, j, (mul,), 1.0)
            for j, ((mul, _), b) in enumerate(zip(cartesian_out, biases))
            if b
        ]
        self.weight_numel = sum(math.prod(ins.path_shape) for ins in paths)
        self.bias_numel = sum(mul for (mul, _), b in zip(cartesian_out, biases) if b)
        for name, size, zeros in (
            ("weight", self.weight_numel, False),
            ("bias", self.bias_numel, True),
        ):
            if self.internal_weights and size:
                self.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.zeros(size) if zeros else torch.randn(size)
                    ),
                )
            else:
                self.register_buffer(name, torch.empty(0))
        self.slices_in = self.irreps_in.slices()
        self.cartesian_out = cartesian_out
        self.projection = (
            ChangeOfBasis(cartesian_out, inverse=True)
            if output_basis == "spherical"
            else Projector(cartesian_out)
            if project
            else torch.nn.Identity()
        )
        connected = {ins.i_out for ins in self.instructions if all(ins.path_shape)}
        self.register_buffer(
            "output_mask",
            (
                torch.cat(
                    [
                        torch.full((item.dim,), float(i in connected))
                        for i, item in enumerate(self.irreps_out)
                    ]
                )
                if self.irreps_out
                else torch.empty(0)
            ),
            persistent=False,
        )

    def forward(self, features, weight=None, bias=None):
        """Apply channel-linear maps.

        Parameters
        ----------
        features : torch.Tensor
            Cartesian features of shape (..., irreps_in.dim).
        weight, bias : torch.Tensor, optional
            External weights and biases with trailing sizes weight_numel and
            bias_numel. Leading dimensions broadcast when shared_weights=False.

        Returns
        -------
        torch.Tensor
            Features of shape (..., irreps_out.dim) in the selected output basis.
        """
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        if weight.shape[-1] != self.weight_numel or bias.shape[-1] != self.bias_numel:
            raise ValueError("Weight or bias size does not match the instructions.")
        if self.shared_weights and (weight.ndim != 1 or bias.ndim != 1):
            raise ValueError("Shared weights and biases must be one-dimensional.")
        shape = torch.broadcast_shapes(
            features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
        )
        zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
        outputs = [
            features.new_zeros((*shape, mul, ir.dim)) + zero
            for mul, ir in self.cartesian_out
        ]
        offset, bias_offset = 0, 0
        for ins in self.instructions:
            size = math.prod(ins.path_shape)
            if ins.i_in < 0:
                value = bias[..., bias_offset : bias_offset + size].unsqueeze(-1)
                bias_offset += size
            else:
                mul, ir = self.irreps_in[ins.i_in]
                x = features[..., self.slices_in[ins.i_in]].reshape(
                    *features.shape[:-1], mul, ir.dim
                )
                w = weight[..., offset : offset + size].reshape(
                    *weight.shape[:-1], *ins.path_shape
                )
                value = torch.einsum("...ui,...uv->...vi", x, w) * ins.path_weight
                offset += size
            outputs[ins.i_out] = outputs[ins.i_out] + value
        output = (
            torch.cat([x.flatten(-2) for x in outputs], -1)
            if outputs
            else features.new_zeros((*shape, 0)) + zero
        )
        return self.projection(output)

    def weight_view_for_instruction(self, instruction, weight=None):
        """Return the channel-weight view for a weighted instruction."""
        ins = self.instructions[instruction]
        if ins.i_in < 0:
            raise ValueError("Bias instructions do not have channel weights.")
        weight = self.weight if weight is None else weight
        offset = sum(
            math.prod(x.path_shape)
            for x in self.instructions[:instruction]
            if x.i_in >= 0
        )
        return weight[..., offset : offset + math.prod(ins.path_shape)].reshape(
            *weight.shape[:-1], *ins.path_shape
        )

    def weight_views(self, weight=None, yield_instruction=False):
        """Iterate over weight views in instruction order."""
        for i, ins in enumerate(self.instructions):
            if ins.i_in >= 0:
                view = self.weight_view_for_instruction(i, weight)
                yield (i, ins, view) if yield_instruction else view

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights, output_basis={self.output_basis}"
