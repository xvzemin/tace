"""Equivalent aligned and frame-free tensor-product convolutions."""

import math

import torch
from e3nn import o3

from ...ops import config
from ...ops.precision import validate_module_precision
from ...ops.torch.conv.o2_o3 import convolution as torch_kernel
from ...ops.torch.o2 import wigner as torch_wigner
from ...utils.autograd import forward_ad_active
from ...utils.metadata import parse_metadata


class O2O3TensorProductConv(torch.nn.Module):
    """Evaluate an O(3) convolution through O(2) restriction in ``ir_mul`` layout.

    Parameters
    ----------
    tensor_product : equivariantx.o2.O3TensorProduct
        Tensor product defining paths, normalization and feature layouts.
    backend : {"torch", "cuda"}, optional
        Execution backend. ``"torch"`` evaluates the selected algorithm with
        optimized PyTorch operations and automatic differentiation by default.
        ``"cuda"`` selects native kernels for CUDA tensors.
        Both support ``"uvu"`` and channel-mixing ``"uvw"`` instructions.
    method : {"auto", "baseline", "generator", "recurrence", "cg", "wigner"}, optional
        ``"baseline"`` retains static path-wise expression selection.
        ``"generator"`` uses a Chebyshev expansion of the squared generator.
        ``"recurrence"`` uses a fixed-parity CG recurrence with shared adjoints.
        ``"cg"`` uses sparse CG contractions of harmonic polynomials.
        ``"wigner"`` constructs a frame and contracts its order-zero CG slices.
        ``"auto"`` measures complete CUDA evaluations and caches the fastest
        method. Warm up before compilation or CUDA Graph capture; otherwise
        these use the baseline until a method has been measured in eager mode.
    radial_mode : {"last", "full"}, optional
        ``"last"`` retains hidden MLP activations. ``"full"`` includes all
        radial layers in the derivative program and recomputes activations.
        CUDA eager evaluation keeps ``"full"`` tiled. Compilation exposes
        the selected differentiable Torch expression program.

    Notes
    -----
    Features use flattened ``ir_mul`` storage. Instructions, weights and
    output multiplicities are preserved. Method selection applies to vector
    inputs. Without vectors, supplied Wigner matrices are used directly.
    CUDA supports float32, float64 and higher derivatives. Atomic reductions
    are not bitwise deterministic.

    Examples
    --------
    >>> import torch
    >>> import equivariantx as eqx
    >>> tp = eqx.o2.O3TensorProduct(
    ...     "2x0e", "1o", "2x1o", [(0, 0, 0, "uvu", True)],
    ...     internal_weights=False, shared_weights=False,
    ... )
    >>> layer = eqx.conv.O2O3TensorProductConv(tp, backend="torch", method="cg")
    >>> features = torch.randn(3, 2)
    >>> vectors = torch.randn(4, 3)
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> radial = torch.randn(4, 8)
    >>> projection = torch.randn(8, tp.weight_numel)
    >>> layer(features, radial, projection, None, torch.ones(4, 1),
    ...       edge_index, 3, vectors=vectors).shape
    torch.Size([3, 6])
    """

    def __init__(
        self, tensor_product, *, backend="torch", method="auto", radial_mode="last"
    ):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.channel_mixing = any(
            ins.connection_mode != "uvu" for ins in tensor_product.instructions
        )
        self.backend = backend
        if radial_mode not in ("last", "full"):
            raise ValueError("radial_mode must be last or full.")
        self.radial_mode = radial_mode
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
        from ...co2.spherical import SphericalCoupling

        input_slices = self.irreps_in.slices()
        output_slices = self.irreps_out.slices()
        self.output_sections = tuple(
            (section.start, section.stop - section.start) for section in output_slices
        )
        self.transverse_couplings = torch.nn.ModuleDict()
        self.harmonics = torch.nn.ModuleDict()
        transverse_paths = []
        offset = 0
        self.path_data = []
        self.sparse_paths = []
        self.coefficient_data = {}
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
            cg = cg * (pole * ins.path_weight)
            # Each row and column of the real m2=0 slice has at most one nonzero.
            if (cg.count_nonzero(0) > 1).any() or (cg.count_nonzero(1) > 1).any():
                raise ValueError("Expected a signed-order-diagonal CG slice.")
            cg_name = f"cg_{len(self.path_data)}"
            self.coefficient_data[cg_name] = cg
            self.register_buffer(
                cg_name, cg.to(tensor_product.weight, copy=True), persistent=False
            )
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
                    coefficients.to(tensor_product.weight, copy=True),
                    persistent=False,
                )
                self.coefficient_data[f"full_cg_{name}"] = coefficients
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

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, value in getattr(self, "coefficient_data", {}).items():
            setattr(self, name, value.to(getattr(self, name), copy=True))
        return self

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
            Higher-precision directions use Wigner matrices constructed in
            their own dtype before conversion to the feature dtype.
        method : str, optional
            Override the constructor's evaluation method for this call.
        radial_network : torch.nn.Module, optional
            Sequential network preceding ``projection``. ``radial_mode``
            selects its evaluation schedule; CUDA full mode uses bounded tiles.
            Append a final affine bias to ``projection`` as an extra row.

        Returns
        -------
        torch.Tensor
            Target features of shape ``(num_nodes, irreps_out.dim)`` in the
            declared, unsimplified ``ir_mul`` layout.
        """
        validate_module_precision(
            self,
            (
                features,
                radial,
                projection,
                wigner,
                amplitudes,
                edge_index,
                num_nodes,
                vectors,
                method,
                radial_network,
            ),
        )
        from ...ops.common.radial import cast_radial

        radial = cast_radial(radial, features.dtype)
        projection = projection.to(features)
        amplitudes = amplitudes.to(features)
        if radial_network is not None and (
            self.radial_mode == "last"
            or (
                (self.backend == "torch" or not features.is_cuda)
                and config.TORCH_REFERENCE
            )
            or forward_ad_active()
        ):
            from ...ops.common.radial import radial_features

            radial = radial_features(radial_network, radial, edge_index, projection)
            radial_network = None
        if self.backend == "cuda" and radial_network is None:
            from ...ops.common.radial import align_projection

            radial, projection = align_projection(radial, projection)
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
        reference = (
            self.backend == "torch" or not features.is_cuda
        ) and config.TORCH_REFERENCE
        if vectors is not None and vectors.dtype != features.dtype:
            wigner = (
                torch_wigner.packed(self.frame, vectors, "recursive")
                if reference
                else self.frame.forward_packed(
                    vectors,
                    method="auto"
                    if self.backend == "cuda" and not forward_ad_active()
                    else "recursive",
                )
            ).to(features)
            vectors = None
        if vectors is not None:
            if method == "auto":
                if forward_ad_active():
                    method = "baseline"
                else:
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
                wigner = (
                    torch_wigner.packed(self.frame, vectors, "recursive")
                    if reference
                    else self.frame.forward_packed(
                        vectors,
                        method="auto"
                        if self.backend == "cuda" and not forward_ad_active()
                        else "recursive",
                    )
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
        wigner = wigner.to(features)
        if self.backend == "cuda" and features.is_cuda and not forward_ad_active():
            from ...ops.cuda.conv.o2_o3 import convolution as cuda_kernel

            return cuda_kernel.evaluate(
                self,
                features,
                radial,
                projection,
                wigner,
                amplitudes,
                edge_index,
                num_nodes,
                vectors,
                radial_network,
            )
        return torch_kernel.evaluate(
            self,
            features,
            radial,
            projection,
            wigner,
            amplitudes,
            edge_index,
            num_nodes,
            vectors,
            radial_network,
        )

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
        if self.backend == "cuda" and features.is_cuda and not forward_ad_active():
            from ...ops.cuda.conv.o2_o3 import convolution as cuda_kernel

            return cuda_kernel.transverse(
                self,
                features,
                radial,
                projection,
                amplitudes,
                edge_index,
                num_nodes,
                vectors,
                method,
                radial_network,
            )
        return torch_kernel.transverse(
            self,
            features,
            radial,
            projection,
            amplitudes,
            edge_index,
            num_nodes,
            vectors,
            method,
            radial_network,
        )

    def reference(
        self, features, radial, projection, wigner, amplitudes, edge_index, num_nodes
    ):
        """Rotate, contract order-zero CG slices, rotate back and aggregate."""
        return torch_kernel.reference(
            self,
            features,
            radial,
            projection,
            wigner,
            amplitudes,
            edge_index,
            num_nodes,
        )

    def scatter_outputs(self, outputs, features, target, num_nodes, zero):
        """Aggregate each declared output block without merging path channels."""
        return torch_kernel.scatter_outputs(
            self, outputs, features, target, num_nodes, zero
        )
