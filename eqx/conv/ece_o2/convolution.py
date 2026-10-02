"""Fused radial linear maps and O(2) edge cluster expansions."""

import torch

from ...o2._clebsch_gordan import clebsch_gordan_product
from ...o2.irreps import Irreps
from ...o2.symmetric_contraction import SymmetricContraction
from ..edge import evaluate, evaluate_torch
from ..program import Program
from ..uv_o2.convolution import frame_description, rotate


def linear_program(program, module, features, weight):
    """Encode a bias-free channel map with its path normalization."""
    outputs = []
    for j, (ir, mul) in enumerate(module.irreps_out):
        terms = []
        for ins, (offset, size) in zip(
            module._weight_instructions, module._weight_offsets
        ):
            if ins.i_out != j:
                continue
            start = module._input_slices[ins.i_in].start
            width = module.irreps_in[ins.i_in].mul
            terms.append(
                program.scale(
                    program.matmul(
                        program.slice(features, start, ir.dim * width),
                        program.slice(weight, offset, size),
                        ir.dim,
                        width,
                        mul,
                    ),
                    ins.path_weight,
                )
            )
        outputs.append(program.sum(terms, ir.dim * mul))
    return program.concatenate(outputs)


def contraction_program(program, module, inputs, weight):
    """Encode recursive paths or generalized-tensor contractions."""
    channels = module.num_channels
    output_dim = module._base_irreps_out.dim
    if module.algorithm == "dense":
        outputs = []
        weight_offset = 0
        for order, paths in enumerate(module._paths_by_order, 1):
            num_paths = len(paths)
            if not num_paths:
                continue
            width = module._base_input_dim
            coefficient = (
                getattr(module, f"generalized_cg_{order}")
                .detach()
                .cpu()
                .reshape(-1, num_paths)
            )
            # Retain the weight-first dense contraction order, but omit zeros
            # in the fixed generalized coupling matrix.
            metadata = (
                channels,
                (num_paths, coefficient.shape[0]),
                tuple(
                    ((path, row), float(coefficient[row, path]))
                    for row, path in coefficient.nonzero().tolist()
                ),
            )
            value = program.product(
                (
                    program.slice(weight, weight_offset, num_paths * channels),
                    program.constant(coefficient.shape[0] * channels, 0),
                ),
                metadata,
                1,
            )
            weight_offset += num_paths * channels
            for i in reversed(range(order)):
                rows = output_dim * width**i
                value = program.channel_contract(
                    (value, inputs[i], program.constant(rows * channels, 0)),
                    rows,
                    width,
                    channels,
                )
            outputs.append(value)
        return program.sum(outputs, output_dim * channels)

    outputs = [[] for _ in range(module._base_irreps_out.num_irreps)]
    cache, weight_offset = {}, 0
    for paths, scales in zip(module._paths_by_order, module._path_scales):
        for path, scale in zip(paths, scales):
            first = path.leaves[0]
            value = program.slice(
                inputs[0],
                module._base_input_slices[first].start * channels,
                module._input_irreps[first].dim * channels,
            )
            for i in range(1, len(path.leaves)):
                key = path.leaves[: i + 1], path.intermediates[: i + 1]
                if key not in cache:
                    ir1, ir2, ir = (
                        path.intermediates[i - 1],
                        module._input_irreps[path.leaves[i]],
                        path.intermediates[i],
                    )
                    coefficient = clebsch_gordan_product(
                        torch.eye(ir1.dim, dtype=torch.float64),
                        ir1,
                        torch.eye(ir2.dim, dtype=torch.float64),
                        ir2,
                        ir,
                    ).permute(1, 2, 0)
                    metadata = (
                        channels,
                        (ir1.dim, ir2.dim, ir.dim),
                        tuple(
                            (tuple(index), float(coefficient[tuple(index)]))
                            for index in coefficient.nonzero().tolist()
                        ),
                    )
                    other = program.slice(
                        inputs[i],
                        module._base_input_slices[path.leaves[i]].start * channels,
                        ir2.dim * channels,
                    )
                    cache[key] = program.product(
                        (value, other, program.constant(ir.dim * channels, 0)),
                        metadata,
                        2,
                    )
                value = cache[key]
            dim = path.intermediates[-1].dim
            path_weight = program.slice(weight, weight_offset, channels)
            path_weight = program.gather(path_weight, tuple(range(channels)) * dim)
            outputs[path.output_index].append(
                program.scale(program.binary("mul", value, path_weight), scale)
            )
            weight_offset += channels
    return program.concatenate(
        [
            program.sum(values, ir.dim * channels)
            for ir, values in zip(module._base_irreps_out.expanded(), outputs)
        ]
    )


class EceO2TensorProductConv(torch.nn.Module):
    """Fuse channel mixing, edge expansion, radial weighting and graph reduction.

    Parameters
    ----------
    frame_in, frame_out : o2.LocalFrame
        Input restriction and output lifting.
    radial_linear : o2.UuLinear
        Externally weighted channelwise map with ``path_mode="expand"``.
    linear_up, linear_down : o2.Linear
        Bias-free channel maps before and after the expansion.
    contraction : o2.SymmetricContraction or o2.AsymmetricContraction
        Many-body basis with ``path_mode="sum"``.
    element_dependent : bool, optional
        Multiply radial path weights by source and target element coefficients.

    Notes
    -----
    Parameters remain owned by the caller. ``set_algorithm`` changes only
    contraction order. CUDA evaluates intermediate edge expressions in a
    bounded workspace and generates recursively differentiable adjoints.
    """

    def __init__(
        self,
        frame_in,
        linear_up,
        contraction,
        radial_linear,
        linear_down,
        frame_out,
        *,
        element_dependent=False,
    ):
        super().__init__()
        if radial_linear.path_mode != "expand" or contraction.path_mode != "sum":
            raise ValueError(
                "ECE requires expanded radial paths and summed contraction paths."
            )
        if linear_up.bias_numel or linear_down.bias_numel:
            raise ValueError("ECE channel maps must be bias-free.")
        # Module descriptors are not registered twice as parameter owners.
        object.__setattr__(self, "frame_in", frame_in)
        object.__setattr__(self, "frame_out", frame_out)
        object.__setattr__(self, "radial_linear", radial_linear)
        object.__setattr__(self, "linear_up", linear_up)
        object.__setattr__(self, "linear_down", linear_down)
        object.__setattr__(self, "contraction", contraction)
        self.element_dependent = element_dependent
        self.weight_numel = radial_linear.weight_numel
        self.num_inputs = (
            1
            if isinstance(contraction, SymmetricContraction)
            else contraction.correlation
        )
        if (
            linear_up.irreps_in
            != Irreps((ir, 2 * mul) for ir, mul in frame_in.irreps_out)
            or linear_up.irreps_out
            != Irreps((ir, mul * self.num_inputs) for ir, mul in contraction.irreps_in)
            or radial_linear.irreps_in != contraction.irreps_out
            or linear_down.irreps_in != radial_linear.irreps_out
            or linear_down.irreps_out != frame_out.irreps_out
        ):
            raise ValueError("The representations of successive ECE stages must match.")
        self.programs = {}

    def _apply(self, fn, recurse=True):
        self.programs.clear()
        return super()._apply(fn, recurse=recurse)

    def set_algorithm(self, algorithm):
        """Change evaluation order while retaining the same parameters."""
        self.contraction.set_algorithm(algorithm)
        self.programs.clear()

    @torch.compiler.assume_constant_result
    def build(self, radial_width, wigner_shape):
        """Construct the edge expression for one static tensor layout."""
        program = Program()
        frame_in, frame_out = self.frame_in, self.frame_out
        radial, contraction = self.radial_linear, self.contraction
        angular_out, angular_in = wigner_shape
        specs = (
            ("source", frame_in.irreps_in.dim),
            ("edge", radial_width),
            ("shared", radial_width * self.weight_numel),
            ("shared", self.weight_numel),
            ("shared", self.linear_up.weight_numel),
            ("shared", self.linear_down.weight_numel),
            ("source", contraction.weight_numel),
            ("target", contraction.weight_numel),
            ("edge", 1),
            ("edge", angular_in * angular_out),
            ("edge", angular_in * angular_out),
        )
        values = [
            program.input(i, kind, width) for i, (kind, width) in enumerate(specs)
        ]
        if self.element_dependent:
            values.extend(
                program.input(i, kind, self.weight_numel)
                for i, kind in ((11, "source"), (12, "target"))
            )
        target = program.input(0, "target", frame_in.irreps_in.dim)
        local = [
            rotate(
                program,
                frame_description(frame_in),
                x,
                values[9],
                angular_out,
                angular_in,
            )
            for x in (values[0], target)
        ]
        paired = program.concatenate(local)
        paired = program.gather(
            paired,
            (
                side * frame_in.irreps_out.dim + s.start + a * mul + c
                for (ir, mul), s in zip(
                    frame_in.irreps_out, frame_in.irreps_out.slices()
                )
                for a in range(ir.dim)
                for side in range(2)
                for c in range(mul)
            ),
        )
        up = linear_program(program, self.linear_up, paired, values[4])
        channels = contraction.num_channels
        copies = [
            program.gather(
                up,
                (
                    s.start + (a * self.num_inputs + copy) * channels + c
                    for (ir, _), s in zip(
                        self.linear_up.irreps_out, self.linear_up.irreps_out.slices()
                    )
                    for a in range(ir.dim)
                    for c in range(channels)
                ),
            )
            for copy in range(self.num_inputs)
        ]
        inputs = copies * contraction.correlation if self.num_inputs == 1 else copies
        coefficients = program.binary("mul", values[6], values[7])
        features = contraction_program(program, contraction, inputs, coefficients)
        weights = program.binary(
            "add",
            program.matmul(values[1], values[2], 1, radial_width, self.weight_numel),
            values[3],
        )
        if self.element_dependent:
            weights = program.binary(
                "mul", weights, program.binary("mul", values[11], values[12])
            )
        radial_outputs = []
        for j, (ir, _) in enumerate(radial.irreps_out):
            parts = []
            for ins, (start, _) in zip(radial.instructions, radial._weight_offsets):
                if ins.i_out != j:
                    continue
                ni, no, channels = ins.path_shape
                begin = radial._input_slices[ins.i_in].start
                x = program.gather(
                    features,
                    (
                        begin + (a * ni + u) * channels + c
                        for a in range(ir.dim)
                        for u in range(ni)
                        for _ in range(no)
                        for c in range(channels)
                    ),
                )
                w = program.gather(
                    weights,
                    (
                        start + (u * no + v) * channels + c
                        for _ in range(ir.dim)
                        for u in range(ni)
                        for v in range(no)
                        for c in range(channels)
                    ),
                )
                parts.append(
                    program.scale(program.binary("mul", x, w), ins.path_weight)
                )
            # Multiple input entries of this irrep are concatenated by channel.
            joined = program.concatenate(parts)
            starts, offset = [], 0
            for part in parts:
                width = program.size(part) // ir.dim
                starts.append((offset, width))
                offset += program.size(part)
            radial_outputs.append(
                program.gather(
                    joined,
                    (
                        start + a * width + c
                        for a in range(ir.dim)
                        for start, width in starts
                        for c in range(width)
                    ),
                )
            )
        down = linear_program(
            program, self.linear_down, program.concatenate(radial_outputs), values[5]
        )
        message = rotate(
            program,
            frame_description(frame_out),
            down,
            values[10],
            angular_out,
            angular_in,
            inverse=True,
        )
        cutoff = program.gather(values[8], (0,) * frame_out.irreps_in.dim)
        message = program.binary("mul", message, cutoff)
        return repr((tuple(program.nodes), ((message, 0, "target"),)))

    def forward(
        self,
        node_features,
        radial_features,
        radial_weight,
        radial_bias,
        weight_up,
        weight_down,
        source_weight,
        target_weight,
        edge_index,
        wigner,
        wigner_inv,
        cutoff,
        *,
        radial_source_weight=None,
        radial_target_weight=None,
        backend="auto",
    ):
        """Evaluate a convolution with shared native and CUDA parameter layouts.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features in flattened ``ir_mul`` layout.
        radial_features : torch.Tensor
            Edge radial features, of shape ``(num_edges, radial_width)``.
        radial_weight, radial_bias : torch.Tensor
            Final radial projection of shape ``(radial_width, weight_numel)``
            and bias of shape ``(weight_numel,)`` for ``radial_linear``.
        weight_up, weight_down : torch.Tensor
            Flattened weights of the two channel maps.
        source_weight, target_weight : torch.Tensor
            Node ECE coefficients of shape
            ``(num_nodes, contraction.weight_numel)``.
        edge_index : torch.Tensor
            Source and target indices, of shape ``(2, num_edges)``.
        wigner, wigner_inv : torch.Tensor
            Dense forward and inverse matrices returned by ``o2.WignerD``.
        cutoff : torch.Tensor
            Message envelope of shape ``(num_edges, 1)``.
        radial_source_weight, radial_target_weight : torch.Tensor, optional
            Node radial coefficients of shape ``(num_nodes, weight_numel)``.
            Required when ``element_dependent=True``.
        backend : {"auto", "cuda", "torch"}, optional
            ``"auto"`` selects CUDA for CUDA tensors and PyTorch otherwise.

        Returns
        -------
        torch.Tensor
            Aggregated node features in flattened ``ir_mul`` layout.
        """
        if backend not in ("auto", "torch", "cuda"):
            raise ValueError("backend must be 'auto', 'torch' or 'cuda'.")
        if backend == "auto":
            backend = "cuda" if node_features.is_cuda else "torch"
        if backend == "cuda" and not node_features.is_cuda:
            raise ValueError("The CUDA backend requires CUDA tensors.")
        if self.element_dependent:
            if radial_source_weight is None or radial_target_weight is None:
                raise ValueError("Element-dependent weights require both endpoints.")
        elif radial_source_weight is not None or radial_target_weight is not None:
            raise ValueError("Set element_dependent=True to use element coefficients.")
        if torch.compiler.is_compiling():
            torch._dynamo.mark_static(radial_features, -1)
            torch._dynamo.mark_static(wigner, 1)
            torch._dynamo.mark_static(wigner, 2)
        key = (
            self.contraction.algorithm,
            radial_features.shape[-1],
            tuple(wigner.shape[1:]),
        )
        if key not in self.programs:
            self.programs[key] = self.build(key[1], key[2])
        metadata = self.programs[key]
        inputs = [
            node_features,
            radial_features,
            radial_weight.reshape(1, -1),
            radial_bias.reshape(1, -1),
            weight_up.reshape(1, -1),
            weight_down.reshape(1, -1),
            source_weight,
            target_weight,
            cutoff,
            wigner,
            wigner_inv,
        ]
        if self.element_dependent:
            inputs.extend((radial_source_weight, radial_target_weight))
        function = evaluate if backend == "cuda" else evaluate_torch
        return function(
            metadata, inputs, edge_index[0], edge_index[1], node_features.shape[0]
        )[0]
