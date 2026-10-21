"""Channel-wise spherical tensor products with factorized weights."""

import math

import torch
from e3nn import o3

from .s2grid import S2Grid

__all__ = ["GauntTensorProduct", "MatrixTensorProduct"]


def irrep_layout(irreps):
    """Group equal irreps without mixing their multiplicity indices."""
    indices = {}
    for (mul, ir), section in zip(irreps, irreps.slices()):
        if mul:
            indices.setdefault(ir, []).extend(range(section.start, section.stop))
    multiplicities = {len(value) // ir.dim for ir, value in indices.items()}
    if len(multiplicities) != 1:
        raise ValueError(
            "Each input and output must have uniform, positive multiplicity."
        )
    mul = multiplicities.pop()
    return mul, {
        ir: torch.tensor(indices[ir], dtype=torch.long, device="cpu").reshape(
            mul, ir.dim
        )
        for ir in sorted(indices)
    }


class FactorizedTensorProduct(torch.nn.Module):
    """Shared layout and weight handling for channel-wise products."""

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        irrep_normalization,
        internal_weights,
        shared_weights,
    ):
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.input_dim1 = self.irreps_in1.dim
        self.input_dim2 = self.irreps_in2.dim
        self.output_dim = self.irreps_out.dim
        if any(
            getattr(ir, "t", 1) != 1
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
            for _, ir in irreps
        ):
            raise ValueError("Only spatial O(3) irreps are supported.")
        if irrep_normalization not in ("component", "norm", "none"):
            raise ValueError("irrep_normalization must be component, norm, or none.")
        self.irrep_normalization = irrep_normalization
        self.shared_weights = True if shared_weights is None else shared_weights
        self.internal_weights = (
            self.shared_weights if internal_weights is None else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("Internal weights must be shared.")
        self.layouts = tuple(
            irrep_layout(irreps)
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        mul1, mul2, mul_out = (item[0] for item in self.layouts)
        self.num_channels = mul_out
        if mul_out != max(mul1, mul2) or any(
            mul not in (1, mul_out) for mul in (mul1, mul2)
        ):
            raise ValueError("Input multiplicities must match the output or equal one.")
        self.constants = {}

    def add_constant(self, name, value):
        """Register a constant while retaining its construction precision."""
        self.constants[name] = value
        self.register_buffer(
            name,
            value.to(
                dtype=torch.get_default_dtype()
                if value.is_floating_point()
                else value.dtype,
                device=torch.get_default_device(),
                copy=True,
            ),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, value in self.constants.items():
            self._buffers[name] = value.to(self._buffers[name], copy=True)
        return self

    def build(self):
        """Prepare sector transforms, factor indices and output normalization."""
        self.inputs = []
        offset = 0
        for number, (mul, layout) in enumerate(self.layouts[:2]):
            weights = {
                ir: torch.arange(offset + i * mul, offset + (i + 1) * mul, device="cpu")
                for i, ir in enumerate(layout)
            }
            offset += len(layout) * mul
            sectors = {}
            for ir in layout:
                sectors.setdefault(self.sector(ir), []).append(ir)
            inputs = []
            for sector, irreps in sorted(sectors.items()):
                name = f"input{number + 1}_{len(inputs)}"
                index = torch.cat([layout[ir] for ir in irreps], dim=-1)
                weight_index = torch.cat(
                    [weights[ir][:, None].expand(-1, ir.dim) for ir in irreps], dim=-1
                )
                scale = torch.cat(
                    [
                        torch.full(
                            (ir.dim,),
                            math.sqrt(ir.dim)
                            if self.irrep_normalization == "norm"
                            else 1.0,
                            dtype=torch.float64,
                            device="cpu",
                        )
                        for ir in irreps
                    ]
                )
                self.add_constant(f"{name}_index", index.flatten())
                self.add_constant(f"{name}_weight_index", weight_index.flatten())
                self.add_constant(f"{name}_scale", scale)
                self.add_constant(f"{name}_basis", self.input_basis(irreps))
                inputs.append((name, sector, tuple(irreps), mul, index.shape[-1]))
            self.inputs.append(tuple(inputs))

        mul, layout = self.layouts[2]
        routes = []
        squared_norms = {ir: 0.0 for ir in layout}
        for i, (_, sector1, irreps1, _, _) in enumerate(self.inputs[0]):
            for j, (_, sector2, irreps2, _, _) in enumerate(self.inputs[1]):
                outputs = []
                for ir in layout:
                    if self.sector(ir) != sector1 * sector2:
                        continue
                    squared_norm = sum(
                        self.reduced_coefficient(ir1.l, ir2.l, ir.l) ** 2
                        for ir1 in irreps1
                        for ir2 in irreps2
                        if ir.p == ir1.p * ir2.p
                        and abs(ir1.l - ir2.l) <= ir.l <= ir1.l + ir2.l
                    )
                    if squared_norm > 0:
                        squared_norms[ir] += squared_norm
                        outputs.append(ir)
                if outputs:
                    routes.append((i, j, tuple(outputs)))

        self.routes = []
        mask = torch.zeros(self.irreps_out.dim, dtype=torch.float64, device="cpu")
        for i, j, irreps in routes:
            name = f"output_{len(self.routes)}"
            index = torch.cat([layout[ir] for ir in irreps], dim=-1)
            weight_index = torch.cat(
                [
                    torch.arange(
                        offset + k * mul, offset + (k + 1) * mul, device="cpu"
                    )[:, None].expand(-1, ir.dim)
                    for k, ir in enumerate(irreps)
                ],
                dim=-1,
            )
            offset += len(irreps) * mul
            scales = []
            for ir in irreps:
                scale = 1.0
                if self.irrep_normalization != "none":
                    scale = math.sqrt(
                        (ir.dim if self.irrep_normalization == "component" else 1.0)
                        / squared_norms[ir]
                    )
                scales.extend([scale] * ir.dim)
            mask[index.flatten()] = 1
            self.add_constant(f"{name}_index", index.flatten())
            self.add_constant(f"{name}_weight_index", weight_index.flatten())
            self.add_constant(
                f"{name}_scale", torch.tensor(scales, dtype=torch.float64, device="cpu")
            )
            self.add_constant(f"{name}_basis", self.output_basis(irreps))
            if getattr(self, "method", None) == "coefficients":
                self.add_constant(
                    f"{name}_coupling",
                    self.coupling_tensor(
                        self.inputs[0][i][2], self.inputs[1][j][2], irreps
                    ),
                )
            self.routes.append((i, j, name, index.shape[-1]))
        self.add_constant("output_mask", mask)
        self.inputs = tuple(
            tuple((name, mul, dim) for name, _, _, mul, dim in entries)
            for entries in self.inputs
        )
        del self.layouts
        self.weight_numel = offset
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0), persistent=False)

    def forward(self, x, y, weight=None):
        """Evaluate the tensor product.

        Parameters
        ----------
        x, y : torch.Tensor
            Features of shape ``(..., irreps_in1.dim)`` and
            ``(..., irreps_in2.dim)`` in ``mul_ir`` layout. Leading dimensions
            are broadcast.
        weight : torch.Tensor, optional
            Factor weights of shape ``(weight_numel,)`` when shared, otherwise
            ``(..., weight_numel)``. Defaults to the internal parameters.

        Returns
        -------
        torch.Tensor
            Features of shape ``(..., irreps_out.dim)`` in ``mul_ir`` layout.
        """
        x, y, weight = self.prepare_inputs(x, y, weight)
        shape = x.shape[:-1]
        result = x.new_zeros(shape + (self.output_dim,))
        if not self.routes:
            return result + (x.sum(-1) + y.sum(-1) + weight.sum(-1)).unsqueeze(-1) * 0
        inputs = []
        for value, entries in zip((x, y), self.inputs):
            transformed = []
            for name, mul, dim in entries:
                features = value.index_select(-1, getattr(self, f"{name}_index"))
                factors = weight.index_select(-1, getattr(self, f"{name}_weight_index"))
                features = (features * factors).reshape(shape + (mul, dim))
                features = features * getattr(self, f"{name}_scale")
                transformed.append(
                    self.transform(features, getattr(self, f"{name}_basis"))
                )
            inputs.append(transformed)
        mul = self.num_channels
        for i, j, name, dim in self.routes:
            value = self.product(inputs[0][i], inputs[1][j], name)
            factors = weight.index_select(-1, getattr(self, f"{name}_weight_index"))
            value = value * factors.reshape(shape + (mul, dim))
            value = value * getattr(self, f"{name}_scale")
            result = result.index_add(
                -1, getattr(self, f"{name}_index"), value.reshape(shape + (mul * dim,))
            )
        return result

    def prepare_inputs(self, x, y, weight):
        """Validate features and broadcast leading dimensions."""
        if x.ndim < 1 or x.shape[-1] != self.input_dim1:
            raise ValueError("The first input must end with irreps_in1.dim.")
        if y.ndim < 1 or y.shape[-1] != self.input_dim2:
            raise ValueError("The second input must end with irreps_in2.dim.")
        if weight is None:
            if not self.internal_weights:
                raise ValueError("External weights are required.")
            weight = self.weight
        if weight.ndim < 1 or weight.shape[-1] != self.weight_numel:
            raise ValueError("Weights must end with weight_numel.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        x, y, weight = (
            value.expand(shape + (value.shape[-1],)) for value in (x, y, weight)
        )
        return x, y, weight

    def extra_repr(self):
        return (
            f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}, "
            f"{self.weight_numel} weights"
        )


class GauntTensorProduct(FactorizedTensorProduct):
    """Multiply spherical signals with both spatial parities.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : e3nn.o3.Irreps or str
        Input and output representations. Each has uniform multiplicity after
        grouping equal irreps. Inputs match the output multiplicity or have
        multiplicity one for channel broadcasting.
    irrep_normalization : {"component", "norm", "none"}, optional
        Unit component variance, unit expected squared irrep norm, or the
        unscaled integral of three surface-orthonormal spherical harmonics.
        Variance normalization assumes independent inputs and unit-variance
        factors. Defaults to ``"component"``.
    internal_weights, shared_weights : bool, optional
        Store trainable factors and share them over leading dimensions.
        Sharing defaults to true; internal weights default to shared_weights.
    method : {"quadrature", "coefficients"}, optional
        Exact-degree sphere quadrature or direct coefficient contraction.
        Defaults to ``"quadrature"``.

    Notes
    -----
    Features use e3nn's real basis and ``mul_ir`` layout. Weight factors act
    on each input irrep/channel and each output irrep/channel/sector pair,
    in that order. They do not parameterize independent CG paths. Pass ones
    for the unweighted product. Output components without a coupling are zero.
    Gaunt couplings with odd ``l1 + l2 + l_out`` vanish, including
    ``1o x 1o -> 1e``. Both natural and unnatural spatial parities are supported.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        *,
        irrep_normalization="component",
        internal_weights=None,
        shared_weights=None,
        method="quadrature",
    ):
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            irrep_normalization,
            internal_weights,
            shared_weights,
        )
        if method not in ("quadrature", "coefficients"):
            raise ValueError("method must be quadrature or coefficients.")
        self.method = method
        lmax = max(
            irreps.lmax
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        degree = sum(
            irreps.lmax
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        grid = S2Grid(
            lmax,
            max(1, 2 * lmax, degree),
            normalization="integral",
            quadrature="gauss_legendre",
            dtype=torch.float64,
            device="cpu",
        )
        self.grid_basis = grid.synthesis
        self.grid_analysis = grid.analysis
        self.build()
        del self.grid_basis, self.grid_analysis

    @staticmethod
    def sector(ir):
        return ir.p * (-1) ** ir.l

    @staticmethod
    def reduced_coefficient(l1, l2, l_out):
        if (l1 + l2 + l_out) % 2:
            return 0.0
        cg = o3.wigner_3j(l1, l2, l_out, dtype=torch.float64, device="cpu")
        return float(cg[l1, l2, l_out]) * math.sqrt(
            (2 * l1 + 1) * (2 * l2 + 1) * (2 * l_out + 1) / (4 * math.pi)
        )

    def input_basis(self, irreps):
        return torch.cat(
            [self.grid_basis[:, ir.l**2 : (ir.l + 1) ** 2] for ir in irreps], -1
        ).T

    def output_basis(self, irreps):
        return torch.cat(
            [self.grid_analysis[ir.l**2 : (ir.l + 1) ** 2] for ir in irreps], 0
        ).T

    def coupling_tensor(self, irreps1, irreps2, irreps_out):
        coefficients = torch.zeros(
            sum(ir.dim for ir in irreps1),
            sum(ir.dim for ir in irreps2),
            sum(ir.dim for ir in irreps_out),
            dtype=torch.float64,
            device="cpu",
        )
        a = 0
        for ir1 in irreps1:
            b = 0
            for ir2 in irreps2:
                c = 0
                for ir in irreps_out:
                    if (
                        ir.p == ir1.p * ir2.p
                        and abs(ir1.l - ir2.l) <= ir.l <= ir1.l + ir2.l
                    ):
                        coefficients[
                            a : a + ir1.dim, b : b + ir2.dim, c : c + ir.dim
                        ] = self.reduced_coefficient(ir1.l, ir2.l, ir.l) * o3.wigner_3j(
                            ir1.l, ir2.l, ir.l, dtype=torch.float64, device="cpu"
                        )
                    c += ir.dim
                b += ir2.dim
            a += ir1.dim
        return coefficients

    def transform(self, features, basis):
        return features if self.method == "coefficients" else features @ basis

    def product(self, x, y, name):
        if self.method == "coefficients":
            return torch.einsum(
                "...ci,...cj,ijk->...ck", x, y, getattr(self, f"{name}_coupling")
            )
        return (x * y) @ getattr(self, f"{name}_basis")


class MatrixTensorProduct(FactorizedTensorProduct):
    """Multiply matrix representations of spherical features with spatial parity.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : e3nn.o3.Irreps or str
        Representations with uniform multiplicity after grouping equal irreps.
        Inputs match the output multiplicity or have multiplicity one.
    irrep_normalization : {"component", "norm", "none"}, optional
        Unit component variance, unit expected squared irrep norm, or unscaled
        multiplication in the orthonormal matrix basis. Normalization assumes
        independent inputs and unit-variance factors. Defaults to ``"component"``.
    internal_weights, shared_weights : bool, optional
        Store trainable factors and share them over leading dimensions.
        Sharing defaults to true; internal weights default to shared_weights.
    matrix_degree : int, optional
        Auxiliary angular degree. Matrices have size ``2 * matrix_degree + 1``.
        Defaults to half the largest requested degree, rounded up.
    backend : {"cuda", "torch"}, optional
        Fused CUDA execution or PyTorch operations. CPU inputs use PyTorch.

    Notes
    -----
    Features use e3nn's real basis and ``mul_ir`` layout. Factors act on each
    input irrep/channel and each output irrep/channel/parity pair, in that
    order. Pass ones for unweighted matrix multiplication. Relative CG path
    coefficients are fixed by matrix_degree; individual paths can vanish.
    Changing matrix_degree changes the operator. Both spatial parities and
    antisymmetric couplings are supported. Channels are not mixed.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        *,
        irrep_normalization="component",
        internal_weights=None,
        shared_weights=None,
        matrix_degree=None,
        backend="cuda",
    ):
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            irrep_normalization,
            internal_weights,
            shared_weights,
        )
        lmax = max(
            irreps.lmax
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        matrix_degree = (lmax + 1) // 2 if matrix_degree is None else matrix_degree
        if not isinstance(matrix_degree, int) or 2 * matrix_degree < lmax:
            raise ValueError(
                "matrix_degree must be an integer with 2 * matrix_degree >= lmax."
            )
        self.matrix_degree = matrix_degree
        self.matrix_dim = 2 * matrix_degree + 1
        self.matrix_basis = {
            l: math.sqrt(2 * l + 1)
            * o3.wigner_3j(
                matrix_degree, matrix_degree, l, dtype=torch.float64, device="cpu"
            )
            for l in {ir.l for _, layout in self.layouts for ir in layout}
        }
        self.build()
        del self.matrix_basis
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be 'cuda' or 'torch'.")
        self.backend = backend
        for entries in self.inputs:
            for name, _, _ in entries:
                self.add_constant(
                    f"{name}_scaled_basis",
                    self.constants[f"{name}_scale"][:, None]
                    * self.constants[f"{name}_basis"],
                )
        for _, _, name, _ in self.routes:
            self.add_constant(
                f"{name}_scaled_basis",
                self.constants[f"{name}_basis"] * self.constants[f"{name}_scale"],
            )
        from ..conv.o3.matrix import matrix_program

        self.metadata, self.matrix_buffers = matrix_program(
            self, weight_storage="shared" if self.shared_weights else "edge"
        )

    def forward(self, x, y, weight=None):
        """Evaluate the product with shared or per-sample factor weights."""
        if not x.is_cuda or self.backend == "torch":
            return super().forward(x, y, weight)
        from ..conv.edge import evaluate

        shared_weight = self.weight if weight is None else weight
        x, y, weight = self.prepare_inputs(x, y, weight)
        shape = x.shape[:-1]
        if self.shared_weights:
            weight = shared_weight
        inputs = [value.reshape(-1, value.shape[-1]) for value in (x, y, weight)]
        inputs.extend(
            getattr(self, name).reshape(1, -1) for name in self.matrix_buffers
        )
        index = torch.arange(inputs[0].shape[0], dtype=torch.long, device=x.device)
        return evaluate(self.metadata, inputs, index, index, index.numel())[0].reshape(
            shape + (self.output_dim,)
        )

    @staticmethod
    def sector(ir):
        return ir.p

    def reduced_coefficient(self, l1, l2, l_out):
        cg = o3.wigner_3j(l1, l2, l_out, dtype=torch.float64, device="cpu")
        index = torch.unravel_index(cg.abs().argmax(), cg.shape)
        a, b, c = (int(item) for item in index)
        value = (
            (self.matrix_basis[l1][..., a] @ self.matrix_basis[l2][..., b])
            * self.matrix_basis[l_out][..., c]
        ).sum()
        coefficient = float(value / cg[a, b, c])
        return (
            0.0
            if abs(coefficient) < 64 * torch.finfo(torch.float64).eps
            else coefficient
        )

    def input_basis(self, irreps):
        return torch.cat([self.matrix_basis[ir.l] for ir in irreps], -1).flatten(0, 1).T

    def output_basis(self, irreps):
        return self.input_basis(irreps).T

    def transform(self, features, basis):
        return (features @ basis).reshape(
            features.shape[:-1] + (self.matrix_dim, self.matrix_dim)
        )

    def product(self, x, y, name):
        return (x @ y).flatten(-2) @ getattr(self, f"{name}_basis")
