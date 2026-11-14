"""TACE-facing compile adapter for optional OAM execution-plan inputs."""

from tace.models.compile.wrapper import CompileTensorModel
from .registration import register_runtime_ops


class OAMCompileAdapter(CompileTensorModel):
    """Retain explicit reciprocal indices at the flat compilation boundary."""

    def __init__(self, readout_fn):
        register_runtime_ops()
        super().__init__(readout_fn)

    def _input_keys(self, data):
        keys = super()._input_keys(data)
        if "oam_radial_map" in data:
            keys = (*keys, "oam_radial_representatives", "oam_radial_map")
        return keys
