"""Equivalent aligned and frame-free tensor-product convolutions."""

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from e3nn import o3

from ...utils.metadata import parse_metadata
from ..contraction import adjoint_program, gradient_mask


@dataclass(eq=False)
class _KernelPlan:
    """Tensor-free scheduling metadata shared by compiled calls."""

    path_data: tuple
    weight_numel: int
    has_unweighted: bool
    sparse_paths: tuple
    scalar_paths: tuple = ()
    harmonic_degrees: tuple = ()


@lru_cache(maxsize=256)
def kernel_plan(metadata):
    return _KernelPlan(*parse_metadata(metadata))


@torch.library.custom_op("eqx::contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Keep runtime scheduling and kernel compilation outside tensor tracing."""
    from ...kernels.cuda_graph import convolution

    plan = kernel_plan(metadata)
    results = contraction_fake(metadata, program, source, target, operands)
    terms = parse_metadata(program)

    def run(inputs, outputs):
        source, target, *values = inputs
        for result in outputs:
            result.zero_()
        if not source.numel() or not plan.path_data:
            return
        from .cuda import contract_many

        calls = [
            (
                tuple(role for role, _ in pairs),
                tuple(values[i] for i in mapping),
                tuple(outputs[slot] for _, slot in pairs),
                weighted_only,
            )
            for mapping, weighted_only, pairs in terms
        ]
        contract_many(plan, source, target, calls)

    convolution(
        "o2_o3", (metadata, program), run, source, target, operands, results, terms, 6
    )
    return results


@contraction.register_fake
def contraction_fake(metadata, program, source, target, operands):
    program = parse_metadata(program)
    results = [None] * (1 + max(slot for _, _, pairs in program for _, slot in pairs))
    for mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


def contraction_setup_context(ctx, inputs, output):
    metadata, program, source, target, operands = inputs
    ctx.kernel_metadata = metadata
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, *operands)


def contraction_backward(ctx, grad_outputs):
    source, target, *operands = ctx.saved_tensors
    program, values, destinations = adjoint_program(
        ctx.program,
        operands,
        grad_outputs,
        gradient_mask(operands, ctx.needs_input_grad[4]),
        kernel_plan(ctx.kernel_metadata).has_unweighted,
    )
    gradients = [None] * len(operands)
    if program:
        results = contraction(
            ctx.kernel_metadata,
            repr(program),
            source,
            target,
            values,
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, None, gradients


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)


class O2O3TensorProductConv(torch.nn.Module):
    """Evaluate an O(3) tensor-product convolution through O(2) restriction.

    Parameters
    ----------
    tensor_product : eqx.o2.O3TensorProduct
        Tensor product defining paths, normalization and feature layouts.
    backend : {"torch", "cuda"}, optional
        Execution backend. ``"torch"`` evaluates the selected algorithm with
        PyTorch operations and automatic differentiation. Defaults to ``"cuda"``
        on CUDA inputs and PyTorch operations on CPU.
        The CUDA backend supports ``"uvu"`` instructions only. Use ``"torch"``
        for channel-mixing ``"uvw"`` instructions.
    method : {"auto", "baseline", "generator", "recurrence", "cg", "wigner"}, optional
        ``"baseline"`` retains static path-wise expression selection.
        ``"generator"`` uses a Chebyshev expansion of the squared generator.
        ``"recurrence"`` uses a fixed-parity CG recurrence with shared adjoints.
        ``"cg"`` uses sparse CG contractions of harmonic polynomials.
        ``"wigner"`` constructs a frame and contracts its order-zero CG slices.
        ``"auto"`` measures complete CUDA evaluations and caches the fastest
        method. Warm up before compilation or CUDA Graph capture; otherwise
        these use the baseline until a method has been measured in eager mode.

    Notes
    -----
    Features use flattened ``ir_mul`` storage. Instructions, weights and
    output multiplicities are preserved. Method selection applies to vector
    inputs. Without vectors, supplied Wigner matrices are used directly.
    CUDA supports float32, float64 and higher derivatives. Atomic reductions
    are not bitwise deterministic.
    """

    def __init__(self, tensor_product, *, backend="cuda", method="auto"):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        if backend == "cuda" and any(
            ins.connection_mode != "uvu" for ins in tensor_product.instructions
        ):
            raise NotImplementedError(
                "The CUDA convolution supports only 'uvu' instructions. "
                "Use backend='torch' for other connection modes."
            )
        self.backend = backend
        if method not in (
            "auto",
            "baseline",
            "generator",
            "recurrence",
            "cg",
            "wigner",
        ):
            raise ValueError(
                "method must be auto, baseline, generator, recurrence, cg or wigner."
            )
        self.method = method
        self.selected_method = "baseline"
        self.tuning_results = {}
        self.normalization = tensor_product.normalization
        from ...o2.wigner import WignerD

        self.frame = WignerD(tensor_product.lmax, tensor_product.lmax).to(
            tensor_product.weight
        )
        self.irreps_in = tensor_product.irreps_in1
        self.irreps_out = tensor_product.irreps_out
        self.output_dim = self.irreps_out.dim
        self.instructions = tuple(tensor_product.instructions)
        self.weight_numel = tensor_product.weight_numel
        self.num_harmonics = tensor_product.num_harmonics
        self.degree_offsets = tuple(
            sum((2 * k + 1) ** 2 for k in range(l))
            for l in range(tensor_product.lmax + 1)
        )
        from ...co2 import SphericalCoupling

        input_slices = self.irreps_in.slices()
        output_slices = self.irreps_out.slices()
        self.transverse_couplings = torch.nn.ModuleDict()
        self.harmonics = torch.nn.ModuleDict()
        transverse_paths = []
        offset = 0
        self.path_data = []
        self.sparse_paths = []
        scalar_paths = []
        harmonic_degrees = []
        for ins in self.instructions:
            mul, ir = self.irreps_in[ins.i_in1]
            mul_out, ir_out = self.irreps_out[ins.i_out]
            ir_sh = tensor_product.irreps_in2[ins.i_in2].ir
            weight_offset = offset if ins.has_weight else -1
            if ins.has_weight:
                offset += math.prod(ins.path_shape)
            if not mul or not mul_out or not ins.path_weight:
                continue
            harmonic_degrees.append(ir_sh.l)
            if ir_sh.l == 0:
                scalar_paths.append(len(self.path_data))
            pole = (
                1.0
                if tensor_product.normalization == "norm"
                else math.sqrt(2 * ir_sh.l + 1)
            )
            if tensor_product.normalization == "integral":
                pole /= math.sqrt(4 * math.pi)
            coefficients = o3.wigner_3j(
                ir.l, ir_sh.l, ir_out.l, dtype=torch.float64, device="cpu"
            )
            cg = coefficients[:, ir_sh.l, :]
            # Static coefficients and tensor buffers use the same construction
            # precision, including when the module is later promoted to float64.
            cg = (cg * (pole * ins.path_weight)).to(torch.get_default_dtype())
            # Each row and column of the real m2=0 slice has at most one nonzero.
            if (cg.count_nonzero(0) > 1).any() or (cg.count_nonzero(1) > 1).any():
                raise ValueError("Expected a signed-order-diagonal CG slice.")
            self.register_buffer(f"cg_{len(self.path_data)}", cg, persistent=False)
            path = (
                input_slices[ins.i_in1].start,
                output_slices[ins.i_out].start,
                mul,
                mul_out,
                ir.dim,
                ir_out.dim,
                self.degree_offsets[ir.l],
                self.degree_offsets[ir_out.l],
                weight_offset,
                ins.i_in2,
            )
            self.path_data.append((ins.connection_mode, path))
            self.sparse_paths.append(
                tuple((m, n, float(cg[m, n])) for m, n in cg.nonzero().tolist())
            )
            name = f"{ir.l}_{ir_sh.l}_{ir_out.l}"
            if name not in self.transverse_couplings:
                self.transverse_couplings[name] = SphericalCoupling(
                    ir.l, ir_sh.l, ir_out.l, tensor_product.normalization
                ).to(tensor_product.weight)
                self.register_buffer(
                    f"full_cg_{name}",
                    coefficients.to(tensor_product.weight),
                    persistent=False,
                )
            if str(ir_sh.l) not in self.harmonics:
                self.harmonics[str(ir_sh.l)] = o3.SphericalHarmonics(
                    ir_sh.l, normalize=False, normalization=self.normalization
                )
            transverse_paths.append(
                (
                    path[0],
                    ins.i_in2,
                    path[1],
                    mul,
                    1,
                    ir.dim,
                    ir_sh.dim,
                    ir_out.dim,
                    weight_offset,
                    ins.path_weight,
                    tuple(
                        (a, b, c, float(coefficients[a, b, c]))
                        for a, b, c in coefficients.nonzero().tolist()
                    ),
                )
            )

        self.has_unweighted = any(path[8] < 0 for _, path in self.path_data)
        self.kernel_metadata = repr(
            (
                tuple(self.path_data),
                self.weight_numel,
                self.has_unweighted,
                tuple(self.sparse_paths),
            )
        )
        self.direction_metadata = repr(
            (
                *parse_metadata(self.kernel_metadata),
                tuple(scalar_paths),
                tuple(harmonic_degrees),
            )
        )
        self.transverse_paths = tuple(transverse_paths)
        self.transverse_layout = tuple((mode, path[3]) for mode, path in self.path_data)
        self.transverse_metadata = repr(
            (
                self.transverse_paths,
                self.weight_numel,
                self.has_unweighted,
                ("transverse", tensor_product.normalization),
            )
        )
        self.generator_metadata = repr(
            (
                self.transverse_paths,
                self.weight_numel,
                self.has_unweighted,
                ("transverse", self.normalization, "generator"),
            )
        )
        self.recurrence_metadata = repr(
            (
                self.transverse_paths,
                self.weight_numel,
                self.has_unweighted,
                ("transverse", self.normalization, "recurrence"),
            )
        )
        self.cg_metadata = repr(
            (
                self.transverse_paths,
                self.weight_numel,
                self.has_unweighted,
                self.normalization,
            )
        )

    def forward(
        self,
        features,
        radial,
        projection,
        wigner,
        amplitudes,
        edge_index,
        num_nodes,
        *,
        vectors=None,
        method=None,
        radial_network=None,
    ):
        """Gather, couple and sum features at target nodes.

        Parameters
        ----------
        features : torch.Tensor
            Node features of shape ``(nodes, irreps_in.dim)`` in flattened
            ``ir_mul`` order.
        radial : torch.Tensor or tuple
            Radial features of shape ``(edges, channels)`` or
            ``(1, channels)``. With an empty projection, these are the
            tensor-product path weights in instruction order.
            With ``radial_network``, ``(tensor, kind)`` partitions may use
            ``kind="edge"``, ``"source"`` or ``"target"``.
        projection : torch.Tensor
            Radial projection of shape ``(channels, weight_numel)``. Use
            shape ``(0, weight_numel)`` for directly supplied path weights.
        wigner : torch.Tensor or None
            Packed degree-wise Wigner matrices, with shape
            ``(edges, sum((2*l+1)**2))`` or a shared leading dimension of one.
            Used when vectors are absent. With vectors, each method evaluates
            its own geometry, including Wigner construction when requested.
        amplitudes : torch.Tensor
            Invariant harmonic amplitudes of shape ``(edges, num_harmonics)``
            or ``(1, num_harmonics)``.
        edge_index : torch.Tensor
            Source and target indices of shape ``(2, edges)``.
        num_nodes : int
            Number of target nodes.
        vectors : torch.Tensor, optional
            Nonzero frame directions, with shape ``(edges, 3)`` or ``(1, 3)``.
            Without vectors, matrices remain independent differentiable inputs.
        method : str, optional
            Override the constructor's evaluation method for this call.
        radial_network : torch.nn.Module, optional
            Sequential network preceding ``projection``, evaluated in bounded
            CUDA tiles. Hidden activations are recomputed during backward.
            A final affine bias may be appended to ``projection`` as a row.

        Returns
        -------
        torch.Tensor
            Target features of shape ``(num_nodes, irreps_out.dim)`` in the
            declared, unsimplified ``ir_mul`` layout.
        """
        if radial_network is not None and (
            self.backend != "cuda" or not features.is_cuda
        ):
            from ..network import materialize

            radial = radial_network(materialize(radial, edge_index))
            if radial.shape[-1] + 1 == projection.shape[0]:
                radial = torch.cat((radial, torch.ones_like(radial[:, :1])), -1)
            radial_network = None
        method = getattr(self, "method", "baseline") if method is None else method
        if method not in (
            "auto",
            "baseline",
            "generator",
            "recurrence",
            "cg",
            "wigner",
        ):
            raise ValueError(
                "method must be auto, baseline, generator, recurrence, cg or wigner."
            )
        if vectors is None and method in ("generator", "recurrence", "cg"):
            raise ValueError(f"method={method!r} requires vectors.")
        if vectors is not None:
            if method == "auto":
                from .autotune import select_method

                method = select_method(
                    self,
                    features,
                    radial,
                    projection,
                    amplitudes,
                    edge_index,
                    num_nodes,
                    vectors,
                    radial_network=radial_network,
                )
            if method == "wigner":
                wigner = self.frame.forward_packed(
                    vectors, method="auto" if self.backend == "cuda" else "recursive"
                )
            elif method in ("baseline", "generator", "recurrence", "cg"):
                return self.forward_transverse(
                    features,
                    radial,
                    projection,
                    amplitudes,
                    edge_index,
                    num_nodes,
                    vectors,
                    method=method,
                    radial_network=radial_network,
                )
        if wigner is None:
            raise ValueError("Supply vectors or packed Wigner matrices.")
        if self.backend == "torch" or not features.is_cuda:
            result = self.reference(
                features, radial, projection, wigner, amplitudes, edge_index, num_nodes
            )
            return result if vectors is None else result + vectors.sum() * 0
        # A zero-stride placeholder supplies the output shape without allocating
        # a second node output. It is never read by the forward contraction.
        output = features.new_empty(1).expand(num_nodes, self.output_dim)
        # Both rotations depend on the same tensor. Keep one operand so their
        # adjoints accumulate together, including in recursively transposed calls.
        program = (((0, 1, 2, 3, 3, 4, 5), False, ((6, 0),)),)
        operands = [
            features,
            radial,
            projection,
            wigner,
            amplitudes,
            output,
        ]
        if radial_network is not None:
            from ..network import convolve

            return convolve(
                "o2",
                self.kernel_metadata,
                [features, radial, projection, wigner, wigner, amplitudes, output],
                edge_index,
                radial_network,
            )
        return contraction(
            self.kernel_metadata,
            repr(program),
            edge_index[0],
            edge_index[1],
            operands,
        )[0]

    def forward_transverse(
        self,
        features,
        radial,
        projection,
        amplitudes,
        edge_index,
        num_nodes,
        vectors,
        *,
        method="baseline",
        radial_network=None,
    ):
        """Contract transverse tensors in spherical storage without alignment."""
        if self.backend == "cuda" and features.is_cuda:
            from ...kernels.wigner import alignment_cuda
            from ..o3.convolution import contraction as spherical_contraction

            if torch.compiler.is_compiling():
                torch._dynamo.mark_static(vectors, -1)
            direction = alignment_cuda(repr("normalize"), [vectors])[0]
            operands = [
                features,
                radial,
                projection,
                amplitudes,
                features.new_empty(1).expand(num_nodes, self.output_dim),
                direction,
            ]
            if radial_network is not None:
                from ..network import convolve

                return convolve(
                    "o3",
                    self.transverse_metadata
                    if method == "baseline"
                    else getattr(self, f"{method}_metadata"),
                    operands,
                    edge_index,
                    radial_network,
                )
            return spherical_contraction(
                self.transverse_metadata
                if method == "baseline"
                else getattr(self, f"{method}_metadata"),
                repr(((tuple(range(6)), False, ((4, 0),)),)),
                edge_index[0],
                edge_index[1],
                operands,
            )[0]
        direction = vectors / vectors.norm(dim=-1, keepdim=True)
        source, target = edge_index
        edges = source.numel()
        weights = radial @ projection if projection.numel() else radial
        zero = sum(
            value.sum() * 0
            for value in (features, radial, projection, amplitudes, direction)
        )
        result = features.new_zeros((num_nodes, self.output_dim)) + zero
        for (mode, mul_out), path in zip(self.transverse_layout, self.transverse_paths):
            (
                start,
                harmonic,
                end,
                mul,
                _,
                dim,
                harmonic_dim,
                dim_out,
                weight,
                factor,
                _,
            ) = path
            x = (
                features[source, start : start + dim * mul]
                .reshape(edges, dim, mul)
                .transpose(-1, -2)
            )
            name = f"{(dim - 1) // 2}_{(harmonic_dim - 1) // 2}_{(dim_out - 1) // 2}"
            if method == "cg":
                harmonic_features = self.harmonics[str((harmonic_dim - 1) // 2)](
                    direction
                )
                value = torch.einsum(
                    "eua,eb,abc->ecu",
                    x,
                    harmonic_features,
                    getattr(self, f"full_cg_{name}"),
                )
            else:
                value = self.transverse_couplings[name](
                    x,
                    direction.unsqueeze(-2),
                    method="recurrence" if method == "recurrence" else "chebyshev",
                ).transpose(-1, -2)
            if weight >= 0:
                width = mul if mode == "uvu" else mul * mul_out
                w = weights[:, weight : weight + width]
                value = (
                    value * w.unsqueeze(-2)
                    if mode == "uvu"
                    else value @ w.reshape(-1, mul, mul_out)
                )
            value = value * (factor * amplitudes[:, harmonic : harmonic + 1]).unsqueeze(
                -1
            )
            result[:, end : end + dim_out * mul_out] += (
                value.flatten(-2)
                .new_zeros((num_nodes, dim_out * mul_out))
                .index_add(0, target, value.flatten(-2))
            )
        return result

    def reference(
        self, features, radial, projection, wigner, amplitudes, edge_index, num_nodes
    ):
        """Rotate, contract order-zero CG slices, rotate back and aggregate."""
        source, target = edge_index
        edges = source.numel()
        weights = radial @ projection if projection.numel() else radial
        zero = sum(
            value.sum() * 0
            for value in (features, radial, projection, wigner, amplitudes)
        )
        result = features.new_zeros((num_nodes, self.output_dim)) + zero
        for index, (mode, path) in enumerate(self.path_data):
            start, end, mul, mul_out, dim, dim_out, dstart, dend, weight, harmonic = (
                path
            )
            x = features[source, start : start + dim * mul].reshape(edges, dim, mul)
            din = wigner[:, dstart : dstart + dim * dim].reshape(
                wigner.size(0), dim, dim
            )
            dout = wigner[:, dend : dend + dim_out * dim_out].reshape(
                wigner.size(0), dim_out, dim_out
            )
            value = getattr(self, f"cg_{index}").to(features).T @ (din @ x)
            if weight >= 0:
                if mode == "uvu":
                    value = value * weights[:, None, weight : weight + mul]
                else:
                    value = value @ weights[:, weight : weight + mul * mul_out].reshape(
                        weights.size(0), mul, mul_out
                    )
            value = dout.transpose(-1, -2) @ value
            message = (value * amplitudes[:, harmonic, None, None]).flatten(1)
            result[:, end : end + dim_out * mul_out] += message.new_zeros(
                (num_nodes, dim_out * mul_out)
            ).index_add(0, target, message)
        return result
