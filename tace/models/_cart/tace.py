"""TACE in irreducible Cartesian tensor storage."""

from copy import deepcopy

import torch

from .._e3nn.node import SphericalTensorNodeEmbedding
from .._e3nn.tace import e3nnTACE
from .layers import cartesian_modules
from .local import RotationMatrices


class cartTACE(e3nnTACE):
    """TACE with native Cartesian contractions and spherical-compatible weights.

    Parameters
    ----------
    **kwargs
        Configuration options accepted by :class:`tace.models.e3nnTACE`.
    """

    rank_two_dim = 9

    def __init__(self, **kwargs):
        kwargs = deepcopy(kwargs)
        # Equivalent CGTP algorithms share paths; native storage uses ICTD.
        atomic = kwargs.get("atomic_basis", {})
        kinds = atomic.get("type", "cgtp")
        atomic["type"] = (
            ["cgtp" if kind in ("co3", "o2_cgtp") else kind for kind in kinds]
            if isinstance(kinds, list)
            else ("cgtp" if kinds in ("co3", "o2_cgtp") else kinds)
        )
        product = kwargs.get("product_basis", {})
        kinds = product.get("type", "cgtp")
        product["type"] = (
            ["nonlinear_cgtp" if kind == "nonlinear_ictd" else kind for kind in kinds]
            if isinstance(kinds, list)
            else ("nonlinear_cgtp" if kinds == "nonlinear_ictd" else kinds)
        )
        super().__init__(**kwargs)
        cartesian_modules(self)
        representation = self.representation
        if isinstance(representation.node_embedding, SphericalTensorNodeEmbedding):
            representation.use_o3_angular_basis = True
        if representation.use_local_frame:
            representation.o2_angular_basis = RotationMatrices()
            representation._can_pack_wigner = False
            representation._can_skip_wigner = False
        if (
            hasattr(self, "pair_representation")
            and self.pair_representation.o2_angular_basis is not None
        ):
            self.pair_representation.o2_angular_basis = RotationMatrices()
        self.model_config["_target_"] = "tace.models.cartTACE"
        kinds = self.model_config["product_basis"]["type"]
        self.model_config["product_basis"]["type"] = (
            ["nonlinear_ictd" if kind == "nonlinear_cgtp" else kind for kind in kinds]
            if isinstance(kinds, list)
            else ("nonlinear_ictd" if kinds == "nonlinear_cgtp" else kinds)
        )
        self.to(dtype=torch.get_default_dtype())
