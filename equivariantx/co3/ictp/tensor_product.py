"""Irreducible Cartesian products with analytic trace subtraction."""

import math
from fractions import Fraction
from functools import lru_cache

import torch

from ...ops import config, resolve_backend
from ...ops.common.irreducible_product import coefficients as sparse_coefficients
from ...ops.common.sparse_coefficients import bilinear_tables
from ...ops.common.torch.cartesian import (
    coupling_size,
    literal_coupling,
    normalized_coefficients,
)
from ...ops.precision import validate_module_precision
from ...ops.torch.co3.ictp import tensor_product as torch_kernel
from ..symmetric import (
    SymmetricBasis,
    SymmetricProduct,
    permutation_classes,
    symmetric_basis,
    symmetric_powers,
)
from ..tensor_product_base import TensorProductBase

SYMMETRIC_MATRIX_SIZE = 8192
SPARSE_PRODUCT_SIZE = 262144
LITERAL_PRODUCT_SIZE = 256
ANALYTIC_TRACE_TERMS = 24576


def path_type(l1, l2, l3):
    """Classify the work on the free indices of two STF inputs."""
    if min(l1, l2, l3) < 0 or not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError(
            "Degrees must be nonnegative and satisfy the triangle inequality."
        )
    if l3 <= 1 or l3 == abs(l1 - l2):
        return "direct"
    if l3 == abs(l1 - l2) + 1:
        return "symmetrize"
    return "detrace"


def symmetric_entries(degrees, budget, compact):
    """Return bounded epsilon coefficients averaged only over free-index permutations."""
    l1, l2, l3 = degrees
    terms = 2 * 3 ** ((l1 + l2 - l3) // 2)
    if 3**l3 * terms > budget:
        return None
    indices, _, _, counts = permutation_classes(l3)
    counts = counts.tolist()
    if not compact and terms * sum(n * n for n in counts) > budget:
        return None
    indices = indices.tolist()
    rows = [[] for _ in counts]
    for a, b, c, scale in normalized_coefficients(*degrees):
        j = indices[c]
        rows[j].append((a, b, scale / counts[j]))
    order = range(len(rows)) if compact else indices
    return tuple(
        (a, b, c, scale) for c, j in enumerate(order) for a, b, scale in rows[j]
    )


def execution(module, features):
    if getattr(module, "backend", "torch") == "cuda" and features.is_cuda:
        from ...ops.cuda.co3.ictp import tensor_product as cuda_kernel

        return cuda_kernel
    return torch_kernel


@lru_cache(maxsize=None)
def identity_expansion(rank, traces):
    """Return the normalized symmetric map for inserting Kronecker deltas."""
    lower = rank - 2 * traces
    powers = symmetric_powers(rank)
    lookup = {p: i for i, p in enumerate(powers)}
    matrix = torch.zeros(
        len(symmetric_powers(lower)), len(powers), dtype=torch.float64, device="cpu"
    )
    for i, q in enumerate(symmetric_powers(lower)):
        multiplicity_in = math.comb(lower, q[0]) * math.comb(lower - q[0], q[1])
        for b in symmetric_powers(traces):
            p = tuple(v + 2 * w for v, w in zip(q, b))
            multiplicity_out = math.comb(rank, p[0]) * math.comb(rank - p[0], p[1])
            multiplicity_delta = math.comb(traces, b[0]) * math.comb(
                traces - b[0], b[1]
            )
            matrix[i, lookup[p]] = (
                math.sqrt(Fraction(multiplicity_in, multiplicity_out))
                * multiplicity_delta
            )
    return matrix


@lru_cache(maxsize=None)
def symmetric_expansion(rank, traces):
    """Combine symmetrization and delta insertion in unique Cartesian storage."""
    lower = rank - 2 * traces
    data = symmetric_basis(lower)
    matrix = torch.zeros(
        3**lower, len(data["norm"]), dtype=torch.float64, device="cpu"
    ).scatter_(
        1,
        data["indices"].unsqueeze(-1),
        data["norm"][data["indices"]].reciprocal().unsqueeze(-1),
    )
    return (matrix @ identity_expansion(rank, traces)) / symmetric_basis(rank)["norm"]


@lru_cache(maxsize=32)
def trace_tables(l1, l2, l3, scales):
    """Collect analytic trace terms by Cartesian exponent triple."""
    entries = []
    out_norm = symmetric_basis(l3)["norm"]
    for t, scale in enumerate(scales):
        lower = l3 - 2 * t
        basis = symmetric_basis(lower)
        matrix = identity_expansion(l3, t) * scale
        matrix = matrix / basis["norm"][:, None] / out_norm
        rows = [
            [(j, value) for j, value in enumerate(row) if value]
            for row in matrix.tolist()
        ]
        index = basis["indices"].tolist()
        for a, b, c, weight in normalized_coefficients(l1, l2, lower):
            entries.extend((a, b, j, weight * v) for j, v in rows[index[c]])
    return bilinear_tables((3**l1, 3**l2, len(out_norm)), tuple(entries))


class CartesianProduct(torch.nn.Module):
    """Couple one pair of STF tensors by symmetric delta and epsilon products.

    Parameters
    ----------
    l1, l2, l3 : int
        Input and output ranks satisfying the triangle inequality.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution provider for the delta/epsilon products.
    """

    def __init__(self, l1, l2, l3, backend=None):
        backend = resolve_backend(backend)
        super().__init__()
        self.backend = backend
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.path_type = path_type(l1, l2, l3)
        k, odd = divmod(l1 + l2 - l3, 2)
        a, b = l1 - k - odd, l2 - k - odd
        self.contractions = k
        if self.path_type == "direct":
            return
        if self.path_type == "symmetrize":
            indices, permutation, offsets, counts = permutation_classes(l3)
            self.register_buffer("indices", indices.clone(), persistent=False)
            self.register_buffer(
                "counts", counts.to(torch.get_default_dtype()), persistent=False
            )
            if backend == "cuda" and l3 > 3:
                self.register_buffer(
                    "permutation", permutation.clone(), persistent=False
                )
                self.register_buffer("offsets", offsets.clone(), persistent=False)
            return
        self.packed = 3**l3 * (l3 + 1) * (l3 + 2) // 2 <= SYMMETRIC_MATRIX_SIZE
        if self.packed:
            self.register_buffer(
                "indices", symmetric_basis(l3)["indices"].clone(), persistent=False
            )
        else:
            self.bases = torch.nn.ModuleList(
                SymmetricBasis(l3 - 2 * t, backend=backend)
                for t in range(min(a, b) + 1)
            )
            if backend == "torch":
                self.symmetric_products = torch.nn.ModuleDict(
                    {
                        str(l3 - 2 * t): SymmetricProduct(l1, l2, l3 - 2 * t)
                        for t in range(min(a, b) + 1)
                        if l3 - 2 * t >= 4
                    }
                )
        coefficient = Fraction(1)
        self.coefficients = [1.0]
        for t in range(1, min(a, b) + 1):
            # Ratio of consecutive trace terms with averaged free-index permutations.
            coefficient *= Fraction(
                -(a - t + 1) * (b - t + 1), t * (2 * l3 - 2 * t + 1)
            )
            # cartesian_product normalizes each additional contraction by sqrt(3).
            scale = float(coefficient) * math.sqrt(3**t)
            self.coefficients.append(scale)
        for t, scale in enumerate(self.coefficients):
            if not t and not self.packed:
                continue
            matrix = (
                symmetric_expansion(l3, t) if self.packed else identity_expansion(l3, t)
            )
            self.register_buffer(
                f"expansion_{t}",
                (matrix * scale).to(torch.get_default_dtype()),
                persistent=False,
            )

        self.sparse_product = (
            self.packed
            and 3 ** (l1 + l2) * (l3 + 1) * (l3 + 2) // 2 <= SPARSE_PRODUCT_SIZE
        )
        self.fused_traces = (
            backend == "cuda"
            and not self.packed
            and sum(
                coupling_size(l1, l2, l3 - 2 * t) * math.comb(t + 2, 2)
                for t in range(len(self.coefficients))
            )
            <= ANALYTIC_TRACE_TERMS
        )
        if self.fused_traces:
            self.sparse_coefficients = trace_tables(
                l1, l2, l3, tuple(self.coefficients)
            )
        elif self.sparse_product:
            self.sparse_coefficients = sparse_coefficients(
                l1,
                l2,
                l3,
                tuple(
                    symmetric_expansion(l3, t) * scale
                    for t, scale in enumerate(self.coefficients)
                ),
            )
            left, right, values = [
                value.tolist() for value in self.sparse_coefficients[-3:]
            ]
            entries = []
            for c, row in enumerate(self.indices.tolist()):
                entries.extend(
                    (a, b, c, coefficient)
                    for a, b, coefficient in zip(left[row], right[row], values[row])
                    if coefficient
                )
                if len(entries) > LITERAL_PRODUCT_SIZE:
                    entries = []
                    break
            self.literal_entries = tuple(entries)
            self.literal_dimensions = (3**l1, 3**l2, 3**l3)
        if self.sparse_product or self.fused_traces:
            for k, value in enumerate(self.sparse_coefficients):
                self.register_buffer(
                    f"sparse_{k}",
                    value.to(dtype=torch.get_default_dtype())
                    if value.is_floating_point()
                    else value.clone(),
                    persistent=False,
                )

    def forward(self, x, y):
        """Return the STF coupling in flattened Cartesian storage."""
        validate_module_precision(self, (x, y))
        return execution(self, x).angular(self, x, y)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if self.path_type == "symmetrize":
            self.counts = permutation_classes(self.l3)[3].to(self.counts)
        if self.path_type != "detrace":
            return self
        for t, scale in enumerate(self.coefficients):
            if not t and not self.packed:
                continue
            name = f"expansion_{t}"
            matrix = (
                symmetric_expansion(self.l3, t)
                if self.packed
                else identity_expansion(self.l3, t)
            )
            setattr(
                self,
                name,
                (matrix * scale).to(getattr(self, name)),
            )
        if self.sparse_product or self.fused_traces:
            for k, value in enumerate(self.sparse_coefficients):
                setattr(
                    self,
                    f"sparse_{k}",
                    value.to(getattr(self, f"sparse_{k}"), copy=True),
                )
        return self


class TensorProduct(TensorProductBase):
    """Couple Cartesian irreps with analytic symmetrization and trace removal.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Input and output representations in flattened ``mul_ir`` layout.
    instructions : sequence of tuple
        ``(i_in1, i_in2, i_out, mode, has_weight[, path_weight])`` entries.
        Modes are ``uvu``, ``uvv``, ``uuu``, ``uuw``, ``uvw``, and ``uvuv``.
    irrep_normalization : {"component", "norm", "none"}, optional
        Normalization of independent irreducible coordinates.
    path_normalization : {"element", "path", "none"}, optional
        Normalization of paths feeding the same output entry.
    in1_var, in2_var, out_var : sequence of float, optional
        Variance for each input or output entry. Defaults to one.
    internal_weights, shared_weights : bool, optional
        Store trainable weights and share weights over leading dimensions.
        Internal weights require shared weights.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Torch or native sparse angular products with analytic trace removal.

    Notes
    -----
    Each path returns a symmetric traceless tensor before summation.
    Even paths use Kronecker deltas. Odd paths also use a Levi-Civita tensor.
    Scalar, vector, and lowest-rank outputs need no free-index correction.
    One rank above the minimum, outputs of rank two or higher need only
    symmetrization. All remaining paths subtract traces on the individual product.
    Path scalars follow the normalization of :class:`equivariantx.co3.TensorProduct`.

    Examples
    --------
    >>> from equivariantx import co3
    >>> tp = co3.ictp.TensorProduct(
    ...     "2x1o", "2x1o", "2x2e", [(0, 0, 0, "uuu", True)]
    ... )
    >>> tp(torch.randn(5, 6), torch.randn(5, 6)).shape
    torch.Size([5, 18])
    """

    @staticmethod
    def coupling_factor(l1, l2, l3):
        """Convert the oriented delta/epsilon product to unit-CG normalization."""
        from ..basis import path_normalization

        return 1.0 / path_normalization(l1, l2, l3)

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        *,
        irrep_normalization="component",
        path_normalization="element",
        in1_var=None,
        in2_var=None,
        out_var=None,
        internal_weights=None,
        shared_weights=None,
        backend=None,
    ):
        backend = resolve_backend(backend)
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            in1_var=in1_var,
            in2_var=in2_var,
            out_var=out_var,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            backend=backend,
        )
        self.project = True
        self.products = torch.nn.ModuleDict()
        couplings = []
        compact = set(range(len(self.irreps_out)))
        for ins in self.instructions:
            degrees = (
                self.irreps_in1[ins.i_in1].ir.l,
                self.irreps_in2[ins.i_in2].ir.l,
                self.irreps_out[ins.i_out].ir.l,
            )
            key = "_".join(map(str, degrees))
            if key not in self.products:
                self.products[key] = CartesianProduct(*degrees, backend=backend)
            couplings.append(degrees)
            if self.products[key].path_type == "direct":
                compact.discard(ins.i_out)
        # Only analytic (already symmetrized) paths may share coordinates.
        # Unreachable outputs need no expansion either.
        compact.intersection_update(ins.i_out for ins in self.instructions)
        self.packed_metadata = (
            self.channel_metadata(couplings, compact) if backend == "cuda" else None
        )
        width, permutation = 0, []
        if self.packed_metadata is not None:
            for i, (mul, ir) in enumerate(self.irreps_out):
                if i in compact:
                    ids = permutation_classes(ir.l)[0].tolist()
                    size = math.comb(ir.l + 2, 2)
                else:
                    ids, size = range(ir.dim), ir.dim
                if compact:
                    permutation.extend(
                        width + ch * size + j for ch in range(mul) for j in ids
                    )
                width += mul * size
        self.packed_dim = width
        self.register_buffer(
            "expansion_index",
            torch.tensor(permutation, dtype=torch.long) if permutation else None,
            persistent=False,
        )

    def channel_entries(self, degrees, budget, compact):
        product = self.products["_".join(map(str, degrees))]
        if product.path_type == "direct":
            return (
                normalized_coefficients(*degrees)
                if literal_coupling(degrees, budget)
                else None
            )
        if product.path_type == "symmetrize":
            return symmetric_entries(degrees, budget, compact)
        if not getattr(product, "sparse_product", False):
            return None
        left, right, values = (v.tolist() for v in product.sparse_coefficients[-3:])
        rows = range(len(left)) if compact else product.indices.tolist()
        entries = []
        for c, row in enumerate(rows):
            entries.extend(
                (a, b, c, v) for a, b, v in zip(left[row], right[row], values[row]) if v
            )
            if len(entries) > budget:
                return None
        return entries

    def forward(self, x, y, weight=None):
        """Couple STF inputs with optional shared or batched channel weights."""
        validate_module_precision(self, (x, y, weight))
        if x.shape[-1] != self.input_dims[0] or y.shape[-1] != self.input_dims[1]:
            raise ValueError("Input feature dimensions do not match the irreps.")
        weight = self.weight if weight is None else weight
        if weight.shape[-1] != self.weight_numel:
            raise ValueError("Weight size does not match the instructions.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        return execution(self, x).evaluate(self, x, y, weight)

    def couple(self, x, y, l1, l2, l3):
        """Evaluate one irreducible angular coupling."""
        product = self.products[f"{l1}_{l2}_{l3}"]
        if getattr(self, "backend", "torch") == "torch" and config.TORCH_REFERENCE:
            return torch_kernel.reference(product, x, y)
        return product(x, y)
