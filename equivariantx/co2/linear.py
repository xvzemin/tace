"""Channel-linear maps for Cartesian O(2) tensors."""

import math

import torch

from equivariantx import o2
from equivariantx.ops.common.torch.linear import linear, linear_layout

from ..ops.precision import validate_module_precision
from ..ops.torch.co2 import linear as torch_kernel
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

    backend : {"torch", "cuda"}, optional
        Execution backend. Native kernels fuse shared channel maps; CPU
        and general broadcasted parameters use Torch.

    Examples
    --------
    >>> from equivariantx import co2
    >>> linear = co2.Linear("4x0e+2x2m", "2x0e+3x2m")
    >>> x = linear.irreps_in.randn(5, -1)
    >>> linear(x).shape
    torch.Size([5, 14])
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
        backend="torch",
    ):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
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
        self.layout = linear_layout(
            [(ir.dim, mul) for mul, ir in self.irreps_in],
            [(ir.dim, mul) for mul, ir in self.cartesian_out],
            self.instructions,
        )
        self.channel_metadata = repr(("mul_ir", self.layout))
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
        validate_module_precision(self, (features, weight, bias))
        if features.shape[-1] != self.input_dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        if weight.shape[-1] != self.weight_numel or bias.shape[-1] != self.bias_numel:
            raise ValueError("Weight or bias size does not match the instructions.")
        if getattr(self, "backend", "torch") == "cuda" and features.is_cuda:
            from ..ops.cuda.co2 import linear as cuda_kernel

            return cuda_kernel.linear(self, features, weight, bias)
        return torch_kernel.linear(self, features, weight, bias)

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

    backend : {"torch", "cuda"}, optional
        Execution backend. Native kernels accept shared or per-node weights.
        CPU tensors and general broadcasts use Torch.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> layer = co2.UuLinear("2x1m", "4x1m", num_channel=2)
    >>> weights = torch.randn(4, layer.weight_numel)
    >>> layer(torch.randn(4, 4), weights).shape
    torch.Size([4, 8])
    """

    def __init__(
        self, irreps_in, irreps_out, num_channel, *, path_mode="sum", backend="torch"
    ):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
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

        from ..ops.common.torch.uu_linear import layout

        self.uu_metadata = layout(
            "mul_ir",
            [(ir.dim, mul) for mul, ir in self.irreps_in],
            [(ir.dim, mul) for mul, ir in self.irreps_out],
            self.instructions,
            num_channel,
            path_mode,
        )

    def forward(self, features, weight):
        """Apply weights ordered as (copies_in, copies_out, num_channel)."""
        validate_module_precision(self, (features, weight))
        if (
            features.shape[-1] != self.irreps_in.dim
            or weight.shape[-1] != self.weight_numel
        ):
            raise ValueError("Incorrect feature or weight dimension.")
        if getattr(self, "backend", "torch") == "cuda" and features.is_cuda:
            from ..ops.cuda.co2 import linear as cuda_kernel

            return cuda_kernel.uu_linear(self, features, weight)
        return torch_kernel.uu_linear(self, features, weight)
