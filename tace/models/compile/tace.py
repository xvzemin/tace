################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from .._e3nn.tace import e3nnTACE as _EagerE3nnTACE


class e3nnTACE(_EagerE3nnTACE):
    def __init__(self, *args, compile_backend: str = "inductor", **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_backend = compile_backend
        self.model_config["compile_backend"] = compile_backend
