################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from ..cartesian.tace import CartesianTACE as _EagerCartesianTACE
from ..spherical.tace import SphericalTACE as _EagerSphericalTACE


class SphericalTACE(_EagerSphericalTACE):
    def __init__(self, *args, compile_backend: str = "inductor", **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_backend = compile_backend
        self.model_config["compile_backend"] = compile_backend


class CartesianTACE(_EagerCartesianTACE):
    def __init__(self, *args, compile_backend: str = "inductor", **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_backend = compile_backend
        self.model_config["compile_backend"] = compile_backend
