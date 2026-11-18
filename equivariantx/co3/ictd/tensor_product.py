"""Cartesian contractions and irreducible projection with independent paths."""

from functools import lru_cache

import torch
from e3nn import o3

from ...ops import config, get_kernel
from ...ops.common.torch.cartesian import (
    cartesian_product,
    literal_coupling,
    normalized_coefficients,
)
from ...ops.precision import validate_module_precision
from ...ops.torch.co3.ictd import tensor_product as torch_kernel
from ..basis import Projector, path_matrix
from ..basis import path_normalization as coupling_scale
from ..irreps import Irreps
from ..symmetric import SymmetricProduct
from ..tensor_product_base import TensorProductBase


@lru_cache(maxsize=None)
def coupling_phase(l1, l2, l3):
    """Return the phase of a Cartesian coupling in the supplied CG convention."""
    cg = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
    indices = tuple(int(i) for i in torch.unravel_index(cg.abs().argmax(), cg.shape))
    x, y, z = [path_matrix(l)[:, m] for l, m in zip((l1, l2, l3), indices)]
    # A single nonzero coefficient determines the phase; the magnitude is analytic.
    value = cartesian_product(x, y, l1, l2, l3) @ z
    return -1.0 if value * cg[indices] < 0 else 1.0


class TensorProduct(TensorProductBase):
    """Couple Cartesian irreps through delta and Levi-Civita contractions.

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
    project : bool, optional
        Project outputs to symmetric traceless tensors. Set False to defer
        projection until after aggregation and channel mixing.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Torch groups compatible paths. CUDA uses sparse angular kernels;
        channel-wise paths share a native product program.

    Notes
    -----
    Inputs must be symmetric traceless. Each path includes the analytic
    Cartesian-to-CG scale. Compatible paths are evaluated together; output
    entries and independent channel weights retain their original ordering.
    Unprojected outputs must be projected before another tensor product or
    a nonlinear operation.

    Examples
    --------
    >>> from equivariantx import co3
    >>> tp = co3.TensorProduct(
    ...     "2x1o", "2x1o", "2x2e", [(0, 0, 0, "uuu", True)]
    ... )
    >>> x, y = torch.randn(5, 6), torch.randn(5, 6)
    >>> tp(x, y).shape
    torch.Size([5, 18])
    """

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
        project=True,
        backend=None,
    ):
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
        self.project = project
        self.projection_indices = tuple(
            sorted(
                {
                    ins.i_out
                    for ins in self.instructions
                    if self.irreps_out[ins.i_out].ir.l > 1
                    and self.irreps_out[ins.i_out].ir.l
                    != abs(
                        self.irreps_in1[ins.i_in1].ir.l
                        - self.irreps_in2[ins.i_in2].ir.l
                    )
                }
            )
        )
        self.projection = Projector(
            self.irreps_out,
            instructions=self.projection_indices if project else (),
            backend=self.backend,
        )
        self.packed_degrees = tuple(
            l
            for l, _ in self.projection.groups
            if l >= 6
            and all(
                i in self.projection_indices
                for i, (_, ir) in enumerate(self.irreps_out)
                if ir.l == l
            )
        )
        self.symmetric_products = torch.nn.ModuleDict(
            {
                "_".join(map(str, d)): SymmetricProduct(*d)
                for d in sorted(set(self.couplings))
                if self.backend == "torch" and d[2] in self.packed_degrees
            }
        )
        self.kernel_metadata = (
            self.channel_metadata(self.couplings)
            if (
                self.backend == "cuda"
                or self.product_degree <= config.CARTESIAN_POLYNOMIAL_DEGREE
            )
            else None
        )

    def forward(self, x, y, weight=None):
        """Evaluate the Cartesian tensor product.

        Parameters
        ----------
        x, y : torch.Tensor
            STF inputs with trailing sizes irreps_in1.dim and irreps_in2.dim.
            Leading dimensions must be broadcastable.
        weight : torch.Tensor, optional
            External weights with trailing size weight_numel. Leading
            dimensions broadcast when shared_weights=False.

        Returns
        -------
        torch.Tensor
            Features of shape (..., irreps_out.dim). Outputs are STF only
            when project=True.
        """
        validate_module_precision(self, (x, y, weight))
        if x.shape[-1] != self.input_dims[0] or y.shape[-1] != self.input_dims[1]:
            raise ValueError("Input feature dimensions do not match the irreps.")
        weight = self.weight if weight is None else weight
        if weight.shape[-1] != self.weight_numel:
            raise ValueError("Weight size does not match the instructions.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
        if self.backend == "cuda" and x.is_cuda:
            from ...ops.cuda.co3.ictd import tensor_product as cuda_kernel

            return cuda_kernel.evaluate(self, x, y, weight)
        return torch_kernel.evaluate(self, x, y, weight)

    def couple(self, x, y, l1, l2, l3):
        """Contract one angular path before output projection."""
        backend = self.backend if x.is_cuda else "torch"
        return get_kernel("cartesian_product", backend)(x, y, l1, l2, l3)

    def symmetric_product(self, x, y, l1, l2, l3):
        basis = self.projection.symmetric_bases[str(l3)]
        if self.backend == "torch":
            return self.symmetric_products[f"{l1}_{l2}_{l3}"](x, y, basis.norm)
        from ...ops.cuda.co3.ictd import tensor_product as cuda_kernel

        return cuda_kernel.symmetric_product(self, x, y, l1, l2, basis)

    @staticmethod
    def coupling_factor(l1, l2, l3):
        return coupling_phase(l1, l2, l3) / coupling_scale(l1, l2, l3)

    def channel_entries(self, degrees, budget, compact):
        """Return bounded unprojected delta and epsilon coefficients."""
        if literal_coupling(degrees, budget):
            return normalized_coefficients(*degrees)


class FullyConnectedTensorProduct(TensorProduct):
    """Connect all allowed irrep triples with independent uvw weights.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : Irreps or str
        Input and output representations.
    **kwargs
        Weight, normalization, and projection options for TensorProduct.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co3
    >>> layer = co3.FullyConnectedTensorProduct("2x1o", "2x1o", "0e + 2e")
    >>> layer(torch.randn(4, 6), torch.randn(4, 6)).shape
    torch.Size([4, 10])
    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        irreps_in1, irreps_in2, irreps_out = map(
            Irreps, (irreps_in1, irreps_in2, irreps_out)
        )
        instructions = [
            (i, j, k, "uvw", True)
            for i, (_, a) in enumerate(irreps_in1)
            for j, (_, b) in enumerate(irreps_in2)
            for k, (_, c) in enumerate(irreps_out)
            if c in a * b
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out, instructions, **kwargs)


class ElementwiseTensorProduct(TensorProduct):
    """Couple corresponding channels without learnable weights.

    Parameters
    ----------
    irreps_in1, irreps_in2 : Irreps or str
        Input representations with equal total multiplicity.
    filter_ir_out : sequence of Irrep or str, optional
        Allowed output irrep labels. Defaults to all allowed couplings.
    **kwargs
        Normalization and projection options for TensorProduct.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co3
    >>> layer = co3.ElementwiseTensorProduct("2x1o", "2x1o", filter_ir_out=["0e"])
    >>> layer(torch.randn(4, 6), torch.randn(4, 6)).shape
    torch.Size([4, 2])
    """

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, **kwargs):
        from ..irreps import Irrep

        a, b = [list(Irreps(irreps).simplify()) for irreps in (irreps_in1, irreps_in2)]
        if sum(mul for mul, _ in a) != sum(mul for mul, _ in b):
            raise ValueError("Elementwise products require equal channel counts.")
        i = 0
        while i < len(a):
            n, ir = a[i]
            m, jr = b[i]
            if n < m:
                b[i : i + 1] = [(n, jr), (m - n, jr)]
            if m < n:
                a[i : i + 1] = [(m, ir), (n - m, ir)]
            i += 1
        keep = None if filter_ir_out is None else set(map(Irrep, filter_ir_out))
        output, instructions = [], []
        for i, ((mul, ir), (_, jr)) in enumerate(zip(a, b)):
            for kr in ir * jr:
                if keep is None or kr in keep:
                    instructions.append((i, i, len(output), "uuu", False))
                    output.append((mul, kr))
        super().__init__(a, b, output, instructions, **kwargs)
