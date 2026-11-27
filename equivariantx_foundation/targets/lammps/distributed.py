"""Eager Kokkos ML-IAP: host halo exchange, rank-local message reductions."""

import logging
from pathlib import Path

import torch

from equivariantx_foundation.models.tace.foundation import frozen_export_copy
from equivariantx_foundation.models.tace.matching import custom_forward
from equivariantx_foundation.models.tace.regional import block_rules
from equivariantx_foundation.registry import require_cuda
from tace.interface.lammps.mliap import TACELammpsCalc
from tace.models.adapter import TensorModel
from tace.models.spherical.representation import NodeRepresentation


def lower_rank_local(model, backend):
    """Install only schedules that do not change the halo or owner contract."""
    rep = getattr(model.readout_fn, "representation", None)
    selected = []
    if type(rep) is not NodeRepresentation or custom_forward(rep):
        return ()
    interactions, products = block_rules(
        model, backend=backend, task="inference", rank_local=True
    )
    for name, match in (("interactions", interactions), ("products", products)):
        layers = getattr(rep, name)
        for i, layer in enumerate(layers):
            select, _ = match(layer)
            lower, _ = select(layer, backend=backend, task="inference")
            if lower is not None:
                layers[i] = lower(layer)
                selected.append(f"{name}.{i}")
    return tuple(selected)


class RegionalDistributedLammpsCalc(TACELammpsCalc):
    """Use the ordinary eager ML-IAP callbacks, including reverse ghost exchange."""

    def _initialize_device(self, data):
        require_cuda(torch.as_tensor(data.elems).device)
        super()._initialize_device(data)
        self.regional_layers = lower_rank_local(self.model.model, backend="cuda")
        if self.regional_layers:
            logging.info("Rank-local EQX layers: %s", self.regional_layers)
        else:
            logging.warning(
                "No eligible rank-local EQX layers; using the host ML-IAP model."
            )


def export_distributed(model, output):
    """Save a CPU-resident loader; select the Kokkos device separately on each rank."""
    rep = getattr(getattr(model, "readout_fn", None), "representation", None)
    if (
        not isinstance(model, TensorModel)
        or type(rep) is not NodeRepresentation
        or any(
            type(m).__module__.startswith("equivariantx_foundation.")
            for m in model.modules()
        )
    ):
        raise ValueError(
            "Distributed ML-IAP requires the unoptimized eager TACE adapter, not a frozen or single-rank schedule."
        )
    output = Path(output)
    # The wrapper sets LAMMPS force flags and freezes its model. Leave any
    # source model, its Parameters, and its optimizer untouched.
    portable = frozen_export_copy(model).cpu()
    calc = RegionalDistributedLammpsCalc(portable)
    torch.save(calc, output)
    return output
