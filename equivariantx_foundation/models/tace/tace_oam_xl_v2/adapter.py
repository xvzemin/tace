"""TACE-facing compile adapter for optional OAM execution-plan inputs."""

import torch

from tace.models.compile.wrapper import CompileTensorModel

from .registration import register_runtime_ops


class OAMCompileAdapter(CompileTensorModel):
    """Retain explicit reciprocal indices at the flat compilation boundary."""

    shape_padding = False

    def __init__(self, readout_fn):
        register_runtime_ops()
        super().__init__(readout_fn)

    def prepare_property_data(self, data):
        """Keep reciprocal-row capacities stable between padded training captures."""
        data = super().prepare_property_data(data)
        rows = data.get("edge_pair_rows")
        if (
            self.training
            and getattr(self.readout_fn, "oam_xl_padded_batches", False)
            and rows is not None
            and rows.numel()
        ):
            # Bind reciprocal capacity to the edge bucket. The margin covers
            # the unpaired dummy atoms; directed graphs retain all their rows.
            capacity = max(
                data["edge_index"].shape[1] // 2 + 128,
                (rows.numel() + 127) // 128 * 128,
            )
            extra = capacity - rows.numel()
            if extra:
                data["edge_pair_rows"] = torch.cat((rows, rows.new_zeros(extra)))
        return data

    def _input_keys(self, data):
        keys = super()._input_keys(data)
        if "oam_radial_map" in data:
            keys = (*keys, "oam_radial_representatives", "oam_radial_map")
        return keys
