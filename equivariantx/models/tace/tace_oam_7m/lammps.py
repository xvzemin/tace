"""Single-rank ML-IAP execution of the five-layer 7M graph."""

from functools import partial

import torch

from ..foundation import export_local_lammps
from ..tace_oam_l.lammps import OAMLammpsCalc
from ..tace_oam_l.receiver import TiledFoundationLammpsCalc
from .model import optimize_oam_7m
from .receiver import SevenReceiverTiles
from .reciprocal import TILE, reuse_pairs
from .tangent import (
    BatchedSeven,
    SevenDistanceTangent,
    projection_capacity,
    retain_projections,
    tangent_embedding,
)


class SevenThroughputLammpsCalc(OAMLammpsCalc):
    """Reuse force projections while they fit the device workspace."""

    def __init__(self, *args, projection_bytes_per_edge):
        super().__init__(*args)
        self.projection_bytes_per_edge = projection_bytes_per_edge

    def _initialize_device(self, data):
        super()._initialize_device(data)
        self.projection_budget = torch.cuda.mem_get_info(self.device)[0] // 2
        self.retaining = None

    def local_forces(self, batch, n, e):
        replay = n <= self.graph_max_atoms and e <= self.graph_max_edges
        rows = ((e + max(8, e // 50) + 63) // 64) * 64 if replay else e
        retain = (
            projection_capacity(rows) * self.projection_bytes_per_edge
            <= self.projection_budget
        )
        if self.retaining != retain:
            self.force_graph = None
            self.retaining = retain
        token = retain_projections.set(retain)
        pair_token = reuse_pairs.set(
            not replay
            and e >= TILE
            and 2 * projection_capacity(rows) * self.projection_bytes_per_edge
            <= self.projection_budget
        )
        try:
            return super().local_forces(batch, n, e)
        finally:
            reuse_pairs.reset(pair_token)
            retain_projections.reset(token)


class SevenCapacityLammpsCalc(TiledFoundationLammpsCalc):
    """Evaluate a compiled 7M graph or its recomputed receiver tiles."""

    model_label = "7M"

    def start_receiver(self, data):
        super().start_receiver(data)
        self.receiver.enable_capacity_tiles()

    @staticmethod
    def build_receiver(model, *, tile, retain_messages):
        receiver = SevenReceiverTiles(model, tile=tile, retain_messages=False)
        receiver.prepare_inference()
        return receiver


def export_lammps(model, output, *, capacity, aoti_package=None):
    """Export a frozen OAM or OMat 7M energy and pair-force graph."""
    frozen = optimize_oam_7m(model)
    rep = frozen.readout_fn.representation
    heads = [
        (i.edge_info.mlp, i.scatter_norm.edge_density.mlp) for i in rep.interactions
    ]
    loader = SevenCapacityLammpsCalc if capacity else OAMLammpsCalc
    if tangent_embedding(rep) is not None:
        rep.__class__ = SevenDistanceTangent
        if not capacity and len({(m[1].eps, m[4].eps) for m, _ in heads}) == 1:
            rep.__class__ = BatchedSeven
            rep.prepare()
            rep.save_projection = True
            rep.reciprocal = True
            bytes_per_edge = (
                sum(i.edge_info.mlp[-1].get_weight().shape[1] for i in rep.interactions)
                * next(rep.parameters()).element_size()
            )
            loader = partial(
                SevenThroughputLammpsCalc, projection_bytes_per_edge=bytes_per_edge
            )
    return export_local_lammps(
        frozen,
        output,
        loader_class=loader,
        receiver_model=frozen if capacity else None,
        aoti_package=aoti_package,
    )
