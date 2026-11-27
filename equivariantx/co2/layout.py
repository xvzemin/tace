"""Irrep feature layouts."""

from ..utils.layout import TransposeIrrepsLayout as BaseTransposeIrrepsLayout
from .irreps import Irreps


class TransposeIrrepsLayout(BaseTransposeIrrepsLayout):
    """Transpose the component and multiplicity axes of Cartesian O(2) features.

    Parameters
    ----------
    irreps : Irreps or str
        Ordered input representation blocks.
    source, target : {"mul_ir", "ir_mul"}
        Input and output component order within each block.
    flat_in, flat_out : bool, optional
        Use one trailing feature axis. Explicit axes require equal multiplicities.
    backend : {None, "torch", "cuda"}, optional
        Execution backend.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> layer = co2.TransposeIrrepsLayout("2x2m", source="mul_ir", target="ir_mul")
    >>> layer(torch.arange(8).float())[:4].tolist()
    [0.0, 4.0, 1.0, 5.0]
    """

    irreps_type = Irreps
