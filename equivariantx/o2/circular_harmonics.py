import torch

from ..ops.torch.o2 import circular_harmonics as torch_kernel
from ..ops.precision import validate_module_precision, validate_precision
from .irreps import Irrep, Irreps


def circular_harmonics(
    input: torch.Tensor,
    mmax: int,
    *,
    normalize: bool = True,
    backend: str = "torch",
) -> torch.Tensor:
    """Evaluate real circular harmonics through order ``mmax``.

    Parameters
    ----------
    input : torch.Tensor
        Two-dimensional vectors with shape ``(..., 2)``.
    mmax : int
        Largest circular order to return.
    normalize : bool, optional
        Normalize each nonzero vector before constructing the harmonics. If
        ``False``, order ``m`` is homogeneous of degree ``m`` in the input.

    backend : {"torch", "cuda"}, optional
        Torch recurrence or native CUDA expression. CPU inputs use Torch.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(..., 1 + 2 * mmax)`` in contiguous order from
        ``0ee`` through ``mmax``. Every positive-order entry stores its real
        cosine and sine components in that order.
    """
    if not isinstance(mmax, int):
        raise TypeError("mmax must be an integer.")
    if mmax < 0:
        raise ValueError("mmax must be non-negative.")
    if input.ndim < 1 or input.shape[-1] != 2:
        raise ValueError(
            "circular_harmonics input must have trailing dimension 2; "
            f"got {tuple(input.shape)}."
        )

    validate_precision((input,))
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be torch or cuda.")
    if backend == "cuda" and input.is_cuda:
        from ..ops.cuda.o2 import circular_harmonics as cuda_kernel

        return cuda_kernel.evaluate(input, mmax, normalize)
    return torch_kernel.evaluate(input, mmax, normalize)


class CircularHarmonics(torch.nn.Module):
    """Construct real O(2) harmonics with optional time-reversal parity.

    Parameters
    ----------
    mmax : int
        Largest circular order in the output.
    normalize : bool, optional
        If ``True``, normalize each input vector so the output depends only on
        direction. If ``False``, return homogeneous circular solid harmonics.
    time_reversal : bool, optional
        If ``True``, treat the input vector as time odd. Order ``m`` then has
        time parity ``(-1)**m``.

    backend : {"torch", "cuda"}, optional
        Execution backend; joint compilation exposes exact tensor algebra.

    Notes
    -----
    The output follows flattened ``ir_mul`` order from ``0ee`` through
    ``mmax`` and has representation metadata in ``irreps_out``.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import o2
    >>> layer = o2.CircularHarmonics(2)
    >>> layer(torch.randn(4, 2)).shape
    torch.Size([4, 5])
    """

    def __init__(
        self,
        mmax: int,
        *,
        normalize: bool = True,
        time_reversal: bool = False,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
        if not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        self.mmax = mmax
        self.normalize = bool(normalize)
        self.time_reversal = bool(time_reversal)
        input_time_parity = -1 if self.time_reversal else 1
        self.irreps_in = Irreps([(Irrep(1, 0, input_time_parity), 1)])
        self.irreps_out = Irreps(
            [(Irrep("0ee"), 1)]
            + [
                (Irrep(order, 0, input_time_parity**order), 1)
                for order in range(1, mmax + 1)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Evaluate circular harmonics for the input vectors.

        Parameters
        ----------
        input : torch.Tensor
            Two-dimensional vectors with shape ``(..., 2)``.

        Returns
        -------
        torch.Tensor
            Harmonics with shape ``(..., irreps_out.dim)``.
        """
        validate_module_precision(self, (input,))
        return circular_harmonics(
            input,
            self.mmax,
            normalize=self.normalize,
            backend=getattr(self, "backend", "torch"),
        )

    def extra_repr(self) -> str:
        return (
            f"mmax={self.mmax}, normalize={self.normalize}, "
            f"time_reversal={self.time_reversal}"
        )
