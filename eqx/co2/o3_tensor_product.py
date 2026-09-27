"""Exact harmonic tensor products through coordinate-free O(2) restriction."""

import math

import torch
from e3nn import o3

from .. import co3
from .restriction import (
    Restriction,
    coupling_coefficients,
    restriction_scale,
    tensor_power,
    transverse_quarter_turn,
)


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

    Notes
    -----
    Each path retains its own weights and output entry. Inputs must be STF
    when input_basis="cartesian". Directions must be nonzero. This PyTorch
    reference uses redundant Cartesian intermediates, not a fused kernel.
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
    ):
        super().__init__()
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
            else co3.Projector(c) if project else torch.nn.Identity()
        )
        degrees = sorted({a[ins.i_in1].ir.l for ins in self.instructions})
        self.restrictions = torch.nn.ModuleDict(
            {str(l): Restriction(l) for l in degrees}
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

    @classmethod
    def from_tensor_product(
        cls,
        tensor_product,
        *,
        normalization="component",
        input_basis="cartesian",
        output_basis="cartesian",
        project=True,
    ):
        """Copy paths, normalized coefficients, and weights from an O(3) tensor product.

        Parameters
        ----------
        tensor_product : torch.nn.Module
            Spherical CGTP or Cartesian ICTP/ICTC TensorProduct with uvu/uvw paths.
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
        return self.projection(self.contract(self.embedding(features), vectors, weight))

    def contract(self, features, vectors, weight=None):
        """Return raw Cartesian path outputs from Cartesian STF input features."""
        if features.shape[-1] != self.input_dim or vectors.shape[-1] != 3:
            raise ValueError("Input feature or vector dimension is incorrect.")
        weight = self.weight if weight is None else weight
        if weight.ndim < 1 or weight.shape[-1] != self.weight_numel:
            raise ValueError("Weights must have trailing size weight_numel.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        direction = (
            vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        ).unsqueeze(-2)
        shape = torch.broadcast_shapes(
            features.shape[:-1], vectors.shape[:-1], weight.shape[:-1]
        )
        zero = features[..., :0].sum() + vectors[..., :0].sum() + weight[..., :0].sum()
        outputs = [
            features.new_zeros((*shape, mul, dim)) + zero
            for mul, dim in self.output_shapes
        ]
        inputs = {}
        for index, mul, l, dim, section in self.input_paths:
            x = features[..., section].reshape(*features.shape[:-1], mul, dim)
            inputs[index] = self.restrictions[str(l)](x, direction)
        offset = 0
        for ins, (l1, l2, l3), coefficients in zip(
            self.instructions, self.paths, self.coefficients
        ):
            value = None
            for m, coefficient in enumerate(coefficients):
                if coefficient == 0.0:
                    continue
                local = inputs[ins.i_in1][m]
                if (l1 + l2 + l3) % 2:
                    local = transverse_quarter_turn(local, direction)
                term = (
                    tensor_power(direction, l3 - m).unsqueeze(-1) * local.unsqueeze(-2)
                ).flatten(-2)
                term = term * (coefficient * ins.path_weight)
                value = term if value is None else value + term
            if value is None:
                value = (
                    features.new_zeros(
                        (*shape, self.input_multiplicities[ins.i_in1], 3**l3)
                    )
                    + zero
                )
            if ins.has_weight:
                size = math.prod(ins.path_shape)
                w = weight[..., offset : offset + size].reshape(
                    *weight.shape[:-1], *ins.path_shape
                )
                offset += size
                value = (
                    torch.einsum("...ud,...uw->...wd", value, w.squeeze(-2))
                    if ins.connection_mode == "uvw"
                    else value * w
                )
            outputs[ins.i_out] = outputs[ins.i_out] + value
        return (
            torch.cat([v.flatten(-2) for v in outputs], -1)
            if outputs
            else features.new_zeros((*shape, 0)) + zero
        )

    def forward_scatter(
        self, features, vectors, edge_index, weight=None, num_nodes=None
    ):
        """Embed nodes, gather edges, sum raw Cartesian paths, then project.

        Features have shape (nodes, irreps_in1.dim); vectors have shape
        (edges, 3). edge_index[0] and edge_index[1] are source and target.
        """
        source, target = edge_index
        num_nodes = features.shape[0] if num_nodes is None else num_nodes
        cartesian = self.embedding(features)
        message = self.contract(cartesian[source], vectors, weight)
        nodes = message.new_zeros((num_nodes, self.output_dim)).index_add(
            0, target, message
        )
        return self.projection(nodes)

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
