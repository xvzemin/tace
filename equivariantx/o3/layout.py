"""Irrep feature layouts."""

from e3nn.o3 import Irreps

from ..utils.layout import TransposeIrrepsLayout as BaseTransposeIrrepsLayout


class TransposeIrrepsLayout(BaseTransposeIrrepsLayout):
    """Transpose the component and multiplicity axes of spherical O(3) features.

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
    >>> from equivariantx import o3
    >>> layer = o3.TransposeIrrepsLayout("2x1o", source="mul_ir", target="ir_mul")
    >>> layer(torch.arange(6).float())[:4].tolist()
    [0.0, 3.0, 1.0, 4.0]
    """

    irreps_type = Irreps
