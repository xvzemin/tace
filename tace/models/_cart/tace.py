"""TACE in irreducible Cartesian tensor storage."""

import torch

from .._e3nn.inter import INTERACTION, O3CgtpInteraction
from .._e3nn.tace import e3nnTACE
from .layers import cartesian_modules


class cartTACE(e3nnTACE):
    """Cartesian TACE with the spherical model's configuration and parameters.

    Parameters
    ----------
    **kwargs
        Model options accepted by :class:`tace.models.e3nnTACE`.
    """

    rank_two_dim = 9

    def __init__(self, **kwargs):
        types = kwargs.get("atomic_basis", {}).get("type", "cgtp")
        types = [types] if isinstance(types, str) else types
        if any(INTERACTION.get(name) is not O3CgtpInteraction for name in types):
            raise NotImplementedError(
                "Cartesian TACE currently implements cgtp interactions only."
            )
        super().__init__(**kwargs)
        cartesian_modules(self)
        self.representation.use_local_frame = False
        self.representation.use_o2 = False
        self.representation.o2_angular_basis = None
        self.model_config["_target_"] = "tace.models.cartTACE"
        self.to(dtype=torch.get_default_dtype())
