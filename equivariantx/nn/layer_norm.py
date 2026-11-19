"""Normalization of direct sums of irreducible representations."""

from math import isfinite

import torch
from e3nn import o3

from ..ops import config, resolve_backend
from ..ops.precision import validate_module_precision
from ..utils.autograd import forward_ad_active


class MergeRmsNorm(torch.nn.Module):
    """Normalize a direct sum of irreps by a shared root mean square.

    Each block contributes its squared norm divided by its multiplicity and
    irreducible dimension. The mean of these contributions sets the scale.

    Parameters
    ----------
    irreps : Irreps or str
        Spherical or Cartesian O(3) or O(2) representations. Strings denote
        spherical O(3) irreps.
    eps : float, optional
        Constant added to the mean square. Defaults to ``1e-5``.
    affine : bool, optional
        Learn one scale per irrep copy.
    layout : {'mul_ir', 'ir_mul'}, optional
        Order within each block. Defaults to ``ir_mul`` for spherical O(2)
        and ``mul_ir`` otherwise.
    backend : {None, 'torch', 'cuda'}, optional
        Execution backend. CPU inputs use PyTorch.

    Examples
    --------
    >>> from equivariantx import nn, co2
    >>> norm = nn.MergeRmsNorm(co2.Irreps("3x0e + 2x2m"))
    >>> norm(torch.randn(5, 11)).shape
    torch.Size([5, 11])
    """

    center_scalars = False

    def __init__(
        self,
        irreps,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        layout=None,
        backend=None,
    ):
        super().__init__()
        from ..o2.irreps import Irrep as O2Irrep
        from ..o2.irreps import Irreps as O2Irreps

        if not hasattr(irreps, "slices"):
            if isinstance(irreps, O2Irrep) or hasattr(irreps, "spherical_dim"):
                irreps = 1 * irreps
            else:
                irreps = o3.Irreps(irreps)
        if not irreps or any(block.mul < 1 for block in irreps):
            raise ValueError("irreps must contain positive multiplicities.")
        if layout is None:
            layout = "ir_mul" if isinstance(irreps, O2Irreps) else "mul_ir"
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError("layout must be 'mul_ir' or 'ir_mul'.")
        if not isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.layout = layout
        self.backend = resolve_backend(backend)
        self.eps = eps
        self.channels = tuple(block.mul for block in irreps)
        labels = tuple(block.ir for block in irreps)
        if isinstance(labels[0], O2Irrep):
            self.dimensions = tuple(1 if ir.m == 0 else 2 for ir in labels)
            scalar = tuple(ir.is_invariant_scalar() for ir in labels)
        else:
            self.dimensions = tuple(2 * ir.l + 1 for ir in labels)
            scalar = tuple(ir.is_scalar() for ir in labels)
        self.scalar_blocks = tuple(
            i for i, yes in enumerate(scalar) if self.center_scalars and yes
        )
        self.scalar_channels = tuple(self.channels[i] for i in self.scalar_blocks)
        self.uniform = len(set(self.channels)) == 1
        self.block_weights = self.default_block_weights
        self.set_storage(irreps)
        self.weight = (
            torch.nn.Parameter(torch.ones(sum(self.channels))) if affine else None
        )
        self.bias = (
            torch.nn.Parameter(torch.zeros(sum(self.scalar_channels)))
            if affine and self.scalar_blocks
            else None
        )

    @property
    def default_block_weights(self):
        """Squared-norm weights per representation block before channel averaging."""
        return tuple(1 / (len(self.dimensions) * dim) for dim in self.dimensions)

    def set_storage(self, irreps):
        """Set the storage representation of the same ordered irrep copies."""
        self.irreps = irreps
        self.components = tuple(block.ir.dim for block in irreps)
        self.widths = tuple(c * d for c, d in zip(self.channels, self.components))
        self.dim = sum(self.widths)

    def inverse_rms(self, features: torch.Tensor) -> torch.Tensor:
        """Return the reciprocal RMS of flat, already centered features."""
        squares = features.square()
        if len(self.widths) == 1:
            variance = squares.mean(-1, keepdim=True) * (
                self.components[0] * self.block_weights[0]
            )
        else:
            variance = sum(
                block.sum(-1, keepdim=True) * (w / c)
                for block, w, c in zip(
                    squares.split(self.widths, dim=-1),
                    self.block_weights,
                    self.channels,
                )
            )
        return (variance + self.eps).rsqrt()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features of shape ``(..., irreps.dim)``, preserving the shape.

        For uniform multiplicities, ``(..., components, channels)`` is also accepted.
        """
        validate_module_precision(self, (features,))
        if features.ndim and features.shape[-1] == self.dim:
            x = features
            channel_last = self.layout == "ir_mul"
        elif (
            self.uniform
            and features.ndim >= 2
            and features.shape[-2:] == (sum(self.components), self.channels[0])
        ):
            x = features.flatten(-2)
            channel_last = True
        else:
            raise ValueError(
                "Expected flat irrep features or uniform (..., components, channels)."
            )
        # A single scalar block has the ordinary channel-wise normalization.
        if (
            not (self.backend == "torch" and config.TORCH_REFERENCE)
            and x.is_cuda
            and (x.dtype == torch.float32 or torch.is_grad_enabled())
            and self.components == (1,)
            and self.block_weights == (1.0,)
            and not forward_ad_active()
        ):
            if self.scalar_blocks:
                out = torch.nn.functional.layer_norm(
                    x, (self.dim,), self.weight, self.bias, self.eps
                )
            else:
                out = torch.nn.functional.rms_norm(
                    x, (self.dim,), self.weight, self.eps
                )
            return out.reshape_as(features)
        native = (
            self.backend == "cuda"
            and x.is_cuda
            and not forward_ad_active()
            and 1 < len(self.widths) <= 16
        )
        if native:
            from ..ops.cuda.nn.layer_norm import normalize

            factors = tuple(w / c for w, c in zip(self.block_weights, self.channels))
            key = repr((self.widths, factors, self.scalar_blocks))
            x = normalize(key, x.reshape(-1, self.dim), self.eps)[0].reshape_as(x)
        elif self.scalar_blocks:
            blocks = list(x.split(self.widths, dim=-1))
            for i in self.scalar_blocks:
                blocks[i] = blocks[i] - blocks[i].mean(-1, keepdim=True)
            x = torch.cat(blocks, dim=-1) if len(blocks) > 1 else blocks[0]

        scale = None if native else self.inverse_rms(x)
        if self.weight is not None:
            gains = [
                (w.expand(d, c) if channel_last else w[:, None].expand(c, d)).reshape(
                    -1
                )
                for w, c, d in zip(
                    self.weight.split(self.channels), self.channels, self.components
                )
            ]
            gain = torch.cat(gains) if len(gains) > 1 else gains[0]
            scale = gain if native else scale * gain
        out = x if scale is None else x * scale
        if self.bias is not None:
            blocks = list(out.split(self.widths, dim=-1))
            for i, bias in zip(
                self.scalar_blocks, self.bias.split(self.scalar_channels)
            ):
                blocks[i] = blocks[i] + bias
            out = torch.cat(blocks, dim=-1) if len(blocks) > 1 else blocks[0]
        return out.reshape_as(features)

    def extra_repr(self) -> str:
        return str(self.irreps)


class MergeLayerNorm(MergeRmsNorm):
    """Center invariant scalar blocks, then normalize the direct sum by a shared RMS.

    Parameters
    ----------
    irreps : Irreps or str
        Input representations as in :class:`MergeRmsNorm`.
    eps : float, optional
        Constant added to the mean square. Defaults to ``1e-5``.
    affine : bool, optional
        Learn one scale per irrep copy and one bias per invariant scalar channel.
    layout : {'mul_ir', 'ir_mul'}, optional
        Order within each block. Defaults to the representation's native layout.
    backend : {None, 'torch', 'cuda'}, optional
        Execution backend. CPU inputs use PyTorch.

    Examples
    --------
    >>> from equivariantx import nn, o2
    >>> norm = nn.MergeLayerNorm(o2.Irreps("4x0e + 2x0o + 3x1m"))
    >>> norm(torch.randn(2, 5, 12)).shape
    torch.Size([2, 5, 12])
    """

    center_scalars = True
