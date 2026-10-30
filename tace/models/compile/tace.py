################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from .._e3nn.tace import e3nnTACE as _EagerE3nnTACE
from .._cart.tace import cartTACE as _EagerCartesianTACE


class e3nnTACE(_EagerE3nnTACE):
    def __init__(self, *args, compile_backend: str = "inductor", **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_backend = compile_backend
        self.model_config["compile_backend"] = compile_backend


class cartTACE(_EagerCartesianTACE):
    def __init__(self, *args, compile_backend: str = "inductor", **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_backend = compile_backend
        self.model_config["compile_backend"] = compile_backend
