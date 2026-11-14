################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from .local import UvConvolution


class MagneticUvConvolution(UvConvolution):
    """Include Cartesian magnetic edge tensors in the local channel map."""

    def forward(
        self,
        features,
        magnetic,
        weights,
        edge_index,
        rotation,
        rotation_inv,
        edge_radial_basis=None,
        edge_cutoff=None,
        radial_network=None,
    ):
        return self.convolution(
            features,
            weights,
            edge_index,
            rotation,
            rotation_inv,
            edge_radial_basis,
            edge_cutoff,
            radial_network,
            magnetic,
        )
