"""Exact harmonic tensor products through coordinate-free O(2) restriction."""

import math

import torch
from e3nn import o3

from equivariantx import co3

from ..ops.precision import validate_module_precision
from ..ops.torch.co2 import o3_tensor_product as torch_kernel
from .restriction import (
    Restriction,
    coupling_coefficients,
    restriction_scale,
)
from .spherical import SphericalCoupling


def execution(module, features):
    if getattr(module, "backend", "torch") == "cuda" and features.is_cuda:
        from ..ops.cuda.co2 import o3_tensor_product as cuda_kernel

        return cuda_kernel
    return torch_kernel


class O3TensorProduct(torch.nn.Module):
    """Couple Cartesian features to implicit three-dimensional harmonics.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        O(3) representation labels. The second input consists of natural-
        parity, time-even unit-direction harmonics with one channel per entry.
    instructions : sequence of tuple
        (i_in1, i_in2, i_out, mode, has_weight[, path_weight]) paths.
        Supported channel modes are uvu and uvw.
    in1_var, in2_var, out_var : sequence of float, optional
        Variance for each irreducible entry. Defaults to one.
    irrep_normalization : {"component", "norm", "none"}, optional
        Irreducible coupling normalization.
    path_normalization : {"element", "path", "none"}, optional
        Normalization across instructions feeding an output entry.
    internal_weights, shared_weights : bool, optional
        Store parameters and share weights over leading dimensions.
    normalization : {"component", "norm", "integral"}, optional
        Spherical-harmonic normalization, folded into fixed coefficients.
    input_basis, output_basis : {"cartesian", "spherical"}, optional
        Input/output storage, both in flattened mul_ir layout.
    project : bool, optional
        Project Cartesian output. Spherical output always includes projection.

    backend : {"torch", "cuda"}, optional
        Execution backend for angular couplings. CPU tensors use Torch.

    Notes
    -----
    Inputs must be STF when input_basis="cartesian". Directions must be nonzero.
    Projected paths use compact spherical coordinates; contract returns the raw
    Cartesian paths and preserves deferred projection.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> layer = co2.O3TensorProduct("2x1o", "1o", "2x0e", [(0, 0, 0, "uvu", True)])
    >>> layer(torch.randn(4, 6), torch.randn(4, 3)).shape
    torch.Size([4, 2])
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        in1_var=None,
        in2_var=None,
        out_var=None,
        irrep_normalization="component",
        path_normalization="element",
        internal_weights=None,
        shared_weights=None,
        *,
        normalization="component",
        input_basis="cartesian",
        output_basis="cartesian",
        project=True,
        backend="torch",
    ):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
        if normalization not in ("component", "norm", "integral"):
            raise ValueError("Unknown harmonic normalization.")
        if input_basis not in ("cartesian", "spherical") or output_basis not in (
            "cartesian",
            "spherical",
        ):
            raise ValueError("Bases must be 'cartesian' or 'spherical'.")
        a, b, c = [co3.Irreps(ir) for ir in (irreps_in1, irreps_in2, irreps_out)]
        if any(mul != 1 or ir.p != (-1) ** ir.l or ir.t != 1 for mul, ir in b):
            raise ValueError(
                "The harmonic input requires one channel, natural parity, and even time parity."
            )
        instructions = list(instructions)
        if any(ins[3] not in ("uvu", "uvw") for ins in instructions):
            raise ValueError("Supported channel modes are uvu and uvw.")
        for i, _, k, mode, has_weight, *_ in instructions:
            if mode == "uvu" and a[i].mul != c[k].mul:
                raise ValueError("uvu requires equal input and output multiplicities.")
            if mode == "uvw" and not has_weight:
                raise ValueError("uvw instructions require weights.")
        metadata = o3.TensorProduct(
            a.spherical(),
            b.spherical(),
            c.spherical(),
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
        self.cartesian_in, self.cartesian_out = a, c
        self.input_dim, self.output_dim = a.dim, c.dim
        self.feature_dim = a.spherical().dim if input_basis == "spherical" else a.dim
        self.input_multiplicities = tuple(mul for mul, _ in a)
        self.output_shapes = tuple((mul, ir.dim) for mul, ir in c)
        self.irreps_in1 = a.spherical() if input_basis == "spherical" else a
        self.irreps_in2 = b
        self.irreps_out = c.spherical() if output_basis == "spherical" else c
        self.input_basis, self.output_basis, self.project = (
            input_basis,
            output_basis,
            project,
        )
        self.normalization = normalization
        self.irrep_normalization = irrep_normalization
        self.path_normalization = path_normalization
        for name in (
            "instructions",
            "weight_numel",
            "internal_weights",
            "shared_weights",
        ):
            setattr(self, name, getattr(metadata, name))
        self.weight_shape = (self.weight_numel,)
        if isinstance(metadata.weight, torch.nn.Parameter):
            self.weight = metadata.weight
        else:
            self.register_buffer("weight", metadata.weight)
        self.embedding = (
            co3.ChangeOfBasis(a) if input_basis == "spherical" else torch.nn.Identity()
        )
        self.projection = (
            co3.ChangeOfBasis(c, inverse=True)
            if output_basis == "spherical"
            else co3.Projector(c, backend=backend)
            if project
            else torch.nn.Identity()
        )
        degrees = sorted({a[ins.i_in1].ir.l for ins in self.instructions})
        self.restrictions = torch.nn.ModuleDict(
            {str(l): Restriction(l, backend=backend) for l in degrees}
        )
        self.paths = tuple(
            (a[ins.i_in1].ir.l, b[ins.i_in2].ir.l, c[ins.i_out].ir.l)
            for ins in self.instructions
        )
        self.coefficients = tuple(
            tuple(
                value * restriction_scale(path[2], m)
                for m, value in enumerate(coupling_coefficients(*path, normalization))
            )
            for path in self.paths
        )
        sections = a.slices()
        self.input_paths = tuple(
            (index, a[index].mul, a[index].ir.l, a[index].ir.dim, sections[index])
            for index in sorted({ins.i_in1 for ins in self.instructions})
        )

        self.compact_embedding = (
            co3.ChangeOfBasis(a, inverse=True)
            if input_basis == "cartesian"
            else torch.nn.Identity()
        )
        self.compact_projection = (
            co3.ChangeOfBasis(c) if output_basis == "cartesian" else torch.nn.Identity()
        )
        self.compact_product = any(max(degrees) > 1 for degrees in self.paths)
        self.couplings = torch.nn.ModuleDict(
            {
                "_".join(map(str, degrees)): SphericalCoupling(
                    *degrees, normalization=normalization, backend=backend
                )
                for degrees in dict.fromkeys(self.paths)
            }
        )
        spherical = a.spherical()
        self.spherical_input_paths = tuple(
            (
                index,
                spherical[index].mul,
                spherical[index].ir.dim,
                spherical.slices()[index],
            )
            for index in sorted({ins.i_in1 for ins in self.instructions})
        )
        self.spherical_output_shapes = tuple((mul, ir.dim) for mul, ir in c.spherical())

    @classmethod
    def from_tensor_product(
        cls,
        tensor_product,
        *,
        normalization="component",
        input_basis="cartesian",
        output_basis="cartesian",
        project=True,
        backend="torch",
    ):
        """Copy paths, normalized coefficients, and weights from an O(3) tensor product.

        Parameters
        ----------
        tensor_product : torch.nn.Module
            Spherical CGTP or Cartesian ICTD TensorProduct with uvu/uvw paths.
        normalization : {"component", "norm", "integral"}, optional
            Harmonic normalization of the original second input.
        input_basis, output_basis : {"cartesian", "spherical"}, optional
            Feature storage of the converted operator.
        project : bool, optional
            Project Cartesian outputs onto their STF subspaces.
        """
        result = cls(
            tensor_product.irreps_in1,
            tensor_product.irreps_in2,
            tensor_product.irreps_out,
            [
                (ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight)
                for ins in tensor_product.instructions
            ],
            internal_weights=tensor_product.internal_weights,
            shared_weights=tensor_product.shared_weights,
            normalization=normalization,
            input_basis=input_basis,
            output_basis=output_basis,
            project=project,
            backend=backend,
        ).to(tensor_product.weight)
        # These path weights already include all variance and normalization factors.
        result.instructions = tuple(tensor_product.instructions)
        result.irrep_normalization = getattr(
            tensor_product, "irrep_normalization", None
        )
        result.path_normalization = getattr(tensor_product, "path_normalization", None)
        with torch.no_grad():
            result.weight.copy_(tensor_product.weight)
        if isinstance(result.weight, torch.nn.Parameter):
            result.weight.requires_grad_(tensor_product.weight.requires_grad)
        return result.train(tensor_product.training)

    def forward(self, features, vectors, weight=None):
        """Return edge features for nonzero (..., 3) vectors and broadcast weights."""
        input_dim = (
            self.feature_dim if hasattr(self, "feature_dim") else self.irreps_in1.dim
        )
        weight = self.check_inputs(features, vectors, weight, input_dim)
        return execution(self, features).forward(self, features, vectors, weight)

    def check_inputs(self, features, vectors, weight, input_dim):
        """Validate public tensor dimensions and resolve shared weights."""
        validate_module_precision(self, (features, vectors, weight))
        if features.shape[-1] != input_dim or vectors.shape[-1] != 3:
            raise ValueError("Input feature or vector dimension is incorrect.")
        weight = self.weight if weight is None else weight
        if weight.ndim < 1 or weight.shape[-1] != self.weight_numel:
            raise ValueError("Weights must have trailing size weight_numel.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        return weight

    def contract(self, features, vectors, weight=None):
        """Return raw Cartesian path outputs from Cartesian STF input features."""
        weight = self.check_inputs(features, vectors, weight, self.input_dim)
        return execution(self, features).contract(self, features, vectors, weight)

    def forward_scatter(
        self, features, vectors, edge_index, weight=None, num_nodes=None
    ):
        """Gather node features, sum edge paths, then expand the output basis.

        Features have shape (nodes, irreps_in1.dim); vectors have shape
        (edges, 3). edge_index[0] and edge_index[1] are source and target.
        """
        input_dim = (
            self.feature_dim if hasattr(self, "feature_dim") else self.irreps_in1.dim
        )
        weight = self.check_inputs(features, vectors, weight, input_dim)
        num_nodes = features.shape[0] if num_nodes is None else num_nodes
        return execution(self, features).scatter(
            self, features, vectors, edge_index, weight, num_nodes
        )

    def weight_view_for_instruction(self, instruction, weight=None):
        """Return an instruction-shaped view of the parameter storage."""
        ins = self.instructions[instruction]
        if not ins.has_weight:
            raise ValueError("The instruction has no weights.")
        weight = self.weight if weight is None else weight
        offset = sum(
            math.prod(p.path_shape)
            for p in self.instructions[:instruction]
            if p.has_weight
        )
        return weight[..., offset : offset + math.prod(ins.path_shape)].reshape(
            *weight.shape[:-1], *ins.path_shape
        )

    def weight_views(self, weight=None, yield_instruction=False):
        """Iterate over weighted paths in their declared order."""
        for i, ins in enumerate(self.instructions):
            if ins.has_weight:
                value = self.weight_view_for_instruction(i, weight)
                yield (i, ins, value) if yield_instruction else value

    def extra_repr(self):
        return f"{self.irreps_in1} x Y({self.irreps_in2}) -> {self.irreps_out} | {self.weight_numel} weights"
