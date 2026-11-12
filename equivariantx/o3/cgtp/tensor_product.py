"""Clebsch--Gordan tensor products with e3nn instructions and normalization."""

import math

import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ...ops import config
from ...ops.kernels.torch.spherical_product import SphericalProduct
from ...ops.precision import validate_module_precision

__all__ = [
    "TensorProduct",
    "FullyConnectedTensorProduct",
    "FullTensorProduct",
    "ElementwiseTensorProduct",
]


class TensorProduct(o3.TensorProduct):
    """Parametrized CG paths with e3nn weights and normalization.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_out : e3nn.o3.Irreps or str
        Two inputs and the output in flattened ``mul_ir`` layout.
    instructions : sequence of tuple
        ``(input1, input2, output, mode, train[, path_weight])`` paths.
        Supports all e3nn connection modes, including triangular pairs.
    backend : {"torch", "cuda"}, optional
        Torch retains the generated GEMM schedule for wide channel maps.
        CUDA uses native sparse angular products followed by channel GEMMs.
    **kwargs
        e3nn variance, normalization, weight and code-generation options.

    Examples
    --------
    >>> tp = TensorProduct("2x1o", "2x1o", "2x0e", [(0, 0, 0, "uuu", True)])
    >>> tp(torch.ones(3, 6), torch.ones(3, 6)).shape
    torch.Size([3, 2])
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        *args,
        backend="torch",
        **kwargs,
    ):
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        super().__init__(
            irreps_in1, irreps_in2, irreps_out, instructions, *args, **kwargs
        )
        self.backend = backend
        self.contraction = SphericalProduct(
            (self.irreps_in1, self.irreps_in2, self.irreps_out),
            self.instructions,
            backend,
        ).to(self.weight)
        self.maximum_channels = max(
            (
                mul
                for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
                for mul, _ in irreps
            ),
            default=0,
        )

    def forward(self, x, y, weight=None):
        """Evaluate ``(..., input_dim)`` features with shared or batched weights."""
        validate_module_precision(self, (x, y, weight))
        if x.shape[-1] != self._in1_dim or y.shape[-1] != self._in2_dim:
            raise ValueError("Input feature dimensions do not match the irreps.")
        weight = self._get_weights(weight)
        validate_module_precision(self, (weight,))
        if self.backend == "cuda":
            if not all(value.is_cuda for value in (x, y, weight)):
                raise ValueError("backend='cuda' requires CUDA inputs and weights.")
            return self.contraction(x, y, weight)
        if (
            not config.TORCH_REFERENCE
            and config.TORCH_GROUPED_CG
            and x.is_cuda
            and 0 < self.maximum_channels <= 16
            and math.prod(x.shape[:-1]) >= 512
            and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
        ):
            return self.contraction(x, y, weight)
        return super().forward(x, y, weight)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for module in self.modules():
            for name, value in tuple(module._buffers.items()):
                if name.startswith("_w3j_"):
                    degrees = tuple(map(int, name[5:].split("_")))
                    setattr(
                        module,
                        name,
                        o3.wigner_3j(*degrees, dtype=torch.float64, device="cpu").to(
                            value
                        ),
                    )
        return self


class FullyConnectedTensorProduct(o3.FullyConnectedTensorProduct, TensorProduct):
    """All admissible weighted CG paths; see :class:`TensorProduct` for options."""


class FullTensorProduct(o3.FullTensorProduct, TensorProduct):
    """All admissible independent channel pairs with TensorProduct options."""


class ElementwiseTensorProduct(o3.ElementwiseTensorProduct, TensorProduct):
    """Corresponding channel pairs with TensorProduct backend and precision rules."""
