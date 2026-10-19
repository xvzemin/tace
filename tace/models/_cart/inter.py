"""Cartesian interaction interfaces."""

from .._e3nn.inter import O3CgtpInteraction


class O2CgtpInteraction(O3CgtpInteraction):
    """Reserved interface for Cartesian local O(2) CGTP interactions."""

    def _setup(self):
        raise NotImplementedError(
            "Cartesian local O(2) interactions are not implemented."
        )


class UvO2Interaction(O2CgtpInteraction):
    """Reserved interface for Cartesian local O(2) channel mixing."""


class UuO2Interaction(O2CgtpInteraction):
    """Reserved interface for Cartesian local O(2) channel-wise maps."""


class UvSO2Interaction(O2CgtpInteraction):
    """Reserved interface for Cartesian local SO(2) channel mixing."""
