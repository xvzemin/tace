"""Irrep feature layouts."""

from ..utils.layout import TransposeIrrepsLayout as BaseTransposeIrrepsLayout
from .irreps import Irreps


class TransposeIrrepsLayout(BaseTransposeIrrepsLayout):
    """Transpose the component and multiplicity axes of Cartesian O(3) features.

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
    >>> from equivariantx import co3
    >>> layer = co3.TransposeIrrepsLayout("2x2e", source="mul_ir", target="ir_mul")
    >>> layer(torch.arange(18).float())[:4].tolist()
    [0.0, 9.0, 1.0, 10.0]
    """

    irreps_type = Irreps
