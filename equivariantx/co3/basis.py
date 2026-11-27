"""Orthonormal path matrices and Cartesian coupling normalization."""

from collections import defaultdict
from functools import lru_cache
from math import factorial, ldexp, prod, sqrt

import torch
from e3nn import o3

from ..ops import resolve_backend
from ..ops.precision import validate_module_precision
from ..ops.torch.co3 import basis as torch_basis
from ..utils.autograd import forward_ad_active
from .irreps import Irreps
from .symmetric import SymmetricBasis


@lru_cache(maxsize=None)
def path_matrix(l):
    """Return the orthonormal path matrix for rank-l STF tensors.

    Parameters
    ----------
    l : int
        Non-negative angular degree.

    Returns
    -------
    torch.Tensor
        Cached CPU float64 matrix of shape (3**l, 2*l+1).
        Do not modify the returned matrix in place.

    Examples
    --------
    >>> from equivariantx import co3
    >>> matrix = co3.path_matrix(2)
    >>> matrix.shape
    torch.Size([9, 5])
    >>> x = torch.randn(4, 5, dtype=matrix.dtype, device=matrix.device)
    >>> tensor = x @ matrix.T
    >>> torch.testing.assert_close(tensor @ matrix, x)
    """
    if l < 0:
        raise ValueError("The angular degree must be non-negative.")
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64, device="cpu")
    cg = o3.wigner_3j(1, l - 1, l, dtype=torch.float64, device="cpu")
    matrix = torch.einsum("an,bnc->bac", path_matrix(l - 1), cg).flatten(0, 1)
    return matrix / matrix.norm(dim=0)


@lru_cache(maxsize=None)
def path_normalization(l1, l2, l3):
    r"""Return the magnitude relating delta/epsilon contractions to unit-norm CG coefficients.

    Parameters
    ----------
    l1, l2, l3 : int
        Input and output degrees satisfying the triangle inequality.

    Returns
    -------
    float
        Positive Cartesian-to-CG scale, excluding the path phase.

    Notes
    -----
    For S = l1+l2+l3 and k = floor((l1+l2-l3)/2), the squared scale is
    ``(S+1)! product((S-2*l)!) / (3**k product((2*l)!))``.
    Integer arithmetic and binary rescaling avoid intermediate overflow.
    """
    if min(l1, l2, l3) < 0 or not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError(
            "Degrees must be non-negative and satisfy the triangle inequality."
        )
    total = l1 + l2 + l3
    numerator = factorial(total + 1) * prod(
        factorial(total - 2 * l) for l in (l1, l2, l3)
    )
    denominator = 3 ** ((l1 + l2 - l3) // 2) * prod(
        factorial(2 * l) for l in (l1, l2, l3)
    )
    exponent = (numerator.bit_length() - denominator.bit_length()) // 2
    if exponent >= 0:
        denominator <<= 2 * exponent
    else:
        numerator <<= -2 * exponent
    return ldexp(sqrt(numerator / denominator), exponent)


class ChangeOfBasis(torch.nn.Module):
    """Convert spherical features and Cartesian tensors using path matrices.

    Parameters
    ----------
    irreps : Irreps or str
        Multiplicities and representation labels, without regrouping.
    inverse : bool, optional
        False maps spherical features to Cartesian tensors. True extracts
        spherical features and discards non-STF components.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Torch matrix operations or compiler-visible native matrix kernels.


    Examples
    --------
    >>> from equivariantx import co3
    >>> to_cartesian = co3.ChangeOfBasis("2x2e")
    >>> to_spherical = co3.ChangeOfBasis("2x2e", inverse=True)
    >>> x = torch.randn(5, 10)
    >>> tensor = to_cartesian(x)
    >>> tensor.shape
    torch.Size([5, 18])
    >>> torch.testing.assert_close(to_spherical(tensor), x)
    """

    def __init__(self, irreps, inverse=False, *, backend=None):
        backend = resolve_backend(backend)
        super().__init__()
        self.backend = backend
        cartesian = Irreps(irreps)
        spherical = cartesian.spherical()
        self.inverse = inverse
        self.irreps_in = cartesian if inverse else spherical
        self.irreps_out = spherical if inverse else cartesian
        self.slices_in = self.irreps_in.slices()
        self.input_dim = self.irreps_in.dim
        self.paths = tuple(
            (mul, ir.l, ir.dim, section)
            for (mul, ir), section in zip(self.irreps_in, self.slices_in)
        )
        groups = []
        for mul, degree, dim, section in self.paths:
            if groups and groups[-1][1] == degree:
                previous = groups.pop()
                groups.append(
                    (
                        previous[0] + mul,
                        degree,
                        dim,
                        slice(previous[3].start, section.stop),
                    )
                )
            else:
                groups.append((mul, degree, dim, section))
        self.basis_groups = tuple(groups)
        for l in sorted({ir.l for _, ir in cartesian}):
            self.register_buffer(
                f"basis_{l}",
                path_matrix(l).to(torch.get_default_dtype()).clone(),
                persistent=False,
            )

    def forward(self, features):
        """Transform features of shape (..., irreps_in.dim) in mul_ir order."""
        validate_module_precision(self, (features,))
        if features.shape[-1] != self.input_dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        if self.backend == "cuda" and features.is_cuda:
            from ..ops.cuda.co3 import basis as cuda_basis

            return cuda_basis.change_basis(self, features)
        return torch_basis.change_basis(self, features)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for l in {path[1] for path in self.paths}:
            name = f"basis_{l}"
            self._buffers[name] = path_matrix(l).to(getattr(self, name)).clone()
        return self

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}, inverse={self.inverse}"


class Projector(torch.nn.Module):
    """Project Cartesian tensors onto their symmetric traceless subspaces.

    Parameters
    ----------
    irreps : Irreps or str
        Representation labels and Cartesian storage layout.
    instructions : sequence of int, optional
        Entries to project. Other entries are passed through unchanged.
        Defaults to all entries. Scalar and vector projections are identities.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CPU tensors use Torch.

    Examples
    --------
    >>> from equivariantx import co3
    >>> project = co3.Projector("2e")
    >>> x = torch.randn(3, 3)
    >>> result = project(x.flatten()).reshape(3, 3)
    >>> expected = (x + x.T) / 2 - torch.eye(3) * x.trace() / 3
    >>> torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    """

    def __init__(self, irreps, instructions=None, *, backend=None):
        backend = resolve_backend(backend)
        super().__init__()
        self.backend = backend
        self.irreps_in = self.irreps_out = Irreps(irreps)
        self.slices = self.irreps_in.slices()
        instructions = (
            range(len(self.irreps_in)) if instructions is None else instructions
        )
        groups = defaultdict(list)
        for i in sorted(set(instructions)):
            if i < 0 or i >= len(self.irreps_in):
                raise IndexError("Projection instruction is outside irreps.")
            mul, ir = self.irreps_in[i]
            if mul and ir.l > 1:
                groups[ir.l].append((i, mul))
        self.groups = tuple(
            (l, tuple(entries)) for l, entries in sorted(groups.items())
        )
        # Keep contiguous channels in one section. On mixed layouts the
        # native rank-two map writes selected STF tensors and passes the rest
        # through, without a gather/cat on either side of its adjoint.
        sections = []
        if len(self.groups) == 1 and self.groups[0][0] == 2:
            for i, _ in self.groups[0][1]:
                s = self.slices[i]
                if sections and sections[-1] == s.start:
                    sections[-1] = s.stop
                else:
                    sections.extend((s.start, s.stop))
        self._rank_two_sections = tuple(sections) if 4 <= len(sections) <= 16 else ()
        self.symmetric_bases = torch.nn.ModuleDict(
            {
                str(l): SymmetricBasis(l, backend=backend)
                for l, _ in self.groups
                if l >= 6
            }
        )
        for l, _ in self.groups:
            self.register_buffer(
                f"basis_{l}",
                self.projection_matrix(l).to(torch.get_default_dtype()).clone(),
                persistent=False,
            )
            if str(l) in self.symmetric_bases and (l + 1) * (l + 2) // 2 <= 128:
                indices, weights = self.parity_blocks(l)
                self.register_buffer(f"columns_{l}", indices.clone(), persistent=False)
                self.register_buffer(
                    f"projector_{l}",
                    weights.to(torch.get_default_dtype()).clone(),
                    persistent=False,
                )

    @classmethod
    @lru_cache(maxsize=32)
    def parity_blocks(cls, l):
        """Return the STF map within coordinate-reflection parity sectors."""
        from .symmetric import symmetric_powers

        basis = cls.projection_matrix(l)
        matrix = basis @ basis.T
        parity = [tuple(k % 2 for k in p) for p in symmetric_powers(l)]
        rows = [[j for j, q in enumerate(parity) if q == p] for p in parity]
        width = max(map(len, rows))
        columns = torch.tensor(
            [row + [0] * (width - len(row)) for row in rows],
            dtype=torch.long,
            device="cpu",
        )
        weights = matrix.gather(1, columns)
        for i, row in enumerate(rows):
            weights[i, len(row) :] = 0.0
        return columns, weights

    @staticmethod
    @lru_cache(maxsize=None)
    def projection_matrix(l):
        """Return CPU float64 projection coefficients in the working layout."""
        matrix = path_matrix(l)
        if l == 2:
            return matrix @ matrix.T
        if l >= 6:
            return SymmetricBasis(l, backend="torch").double().pack(matrix.T).T
        return matrix

    def forward(self, features):
        """Project features of shape (..., irreps_in.dim) in mul_ir order."""
        validate_module_precision(self, (features,))
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        if not self.groups:
            return features
        if (
            self.backend == "cuda"
            and features.is_cuda
            and getattr(self, "_rank_two_sections", ())
            and type(self).forward_blocks is Projector.forward_blocks
            and "forward_blocks" not in vars(self)
            and not forward_ad_active()
        ):
            from ..ops.cuda.co3.basis import project_rank_two_layout

            return project_rank_two_layout(features, self._rank_two_sections)
        values = [
            features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
            for (mul, ir), section in zip(self.irreps_in, self.slices)
        ]
        values = self.forward_blocks(values)
        values = [value.flatten(-2) for value in values]
        if not values:
            return features[..., :0]
        return values[0] if len(values) == 1 else torch.cat(values, dim=-1)

    def forward_blocks(self, features, *, packed_degrees=()):
        """Project blocks of shape (..., mul, 3**l), batching equal ranks."""
        values = list(features)
        validate_module_precision(self, (values,))
        if self.backend == "cuda" and values and values[0].is_cuda:
            from ..ops.cuda.co3 import basis as cuda_basis

            return cuda_basis.project(self, values, packed_degrees=packed_degrees)
        return torch_basis.project(self, values, packed_degrees=packed_degrees)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for l, _ in self.groups:
            name = f"basis_{l}"
            setattr(
                self, name, self.projection_matrix(l).to(getattr(self, name)).clone()
            )
            if hasattr(self, f"projector_{l}"):
                _, matrix = self.parity_blocks(l)
                setattr(
                    self,
                    f"projector_{l}",
                    matrix.to(getattr(self, f"projector_{l}")).clone(),
                )
        return self

    def extra_repr(self):
        return str(self.irreps_in)
