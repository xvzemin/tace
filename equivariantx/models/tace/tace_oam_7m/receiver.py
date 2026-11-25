"""Receiver-tiled derivatives of the released TACE-7M model."""

from typing import NamedTuple

import torch

from equivariantx.models.tace.tace_oam_l.receiver import (
    ReceiverTiles,
    is_cuda_allocation_error,
)

from .model import geometry, radial_heads, radial_parameters, update_node
from .product import FrozenBilinearProduct
from .tangent import distance_inputs, tangent_embedding, tangent_head, tangent_message


class LayerWeights(NamedTuple):
    heads: tuple
    projection: torch.Tensor
    projection_bias: torch.Tensor


class ReceiverWorkspace(NamedTuple):
    """Sizes of the frozen node state, halo features, messages and projection."""

    node: int
    features: int
    message: int
    paths: int

    def resident(self, nodes, pairs):
        # Include one additional full-width allocator block across reverse steps.
        return nodes * (self.node + 4 * self.features + 8) + pairs * 24

    def active(self, n, edges, halo):
        # Include gathered features and their adjoints on distinct source atoms.
        nodes = 6 * self.message * n + 8 * self.features * halo
        # Edge geometry, angular/radial values and adjoints (FP32 lanes).
        edge = 1024 * edges
        # Projected radial values and their derivatives are tiled separately.
        projection = 2 * self.paths * min(32768, edges)
        return 4 * (nodes + edge + projection)

    @staticmethod
    def fits(required, available, total):
        reserve = max(available // 10, 7 * total // 100, 1 << 30)
        return required <= available - reserve


class SevenReceiverTiles(ReceiverTiles):
    """Evaluate TACE-7M atomic energies and pair-energy derivatives.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen spherical TACE-7M adapter.
    tile : int
        Maximum number of receivers per node update.
    retain_messages : bool, optional
        Store the reduced node messages for the reverse pass.
    """

    distance_weight = None
    capacity_limit = None

    def enable_capacity_tiles(self):
        """Size force tiles from live node states and the current graph."""
        if self.distance_weight is None or self.energy_domains is not None:
            return
        self.capacity_limit = 4096
        self.last_capacity_tile = None
        self.workspace = ReceiverWorkspace(
            self.domain_state_bytes([1] * len(self.rep.interactions)),
            max(ir.dim for ir in self.rep.irreps_outs),
            max(i.rejector.eqx_tp.output_dim for i in self.rep.interactions),
            max(r.projection.shape[1] for r in self.radial),
        )

    def plan(self, types, index):
        if self.capacity_limit is None:
            return super().plan(types, index)
        while True:
            self.tile = self.capacity_limit
            tiles = super().plan(types, index)
            # Geometry, COO and tile-local maps are already allocated.
            free, total = torch.cuda.mem_get_info(types.device)
            available = (
                free + torch.cuda.memory_reserved(types.device)
                - torch.cuda.memory_allocated(types.device)
            )
            resident = self.workspace.resident(len(types), index.shape[1])
            active = max(
                (
                    self.workspace.active(len(batch), len(source), len(halo))
                    for _, halo, _, _, source, _, batch in tiles
                ),
                default=0,
            )
            if self.tile == 1024 or self.workspace.fits(
                resident + active, available, total
            ):
                if self.last_capacity_tile != self.tile:
                    print(
                        f"EQX 7M receiver workspace: tile={self.tile}, "
                        f"resident={resident / 2**30:.2f} GiB, "
                        f"active={active / 2**30:.2f} GiB, "
                        f"available={available / 2**30:.2f} GiB",
                        flush=True,
                    )
                    self.last_capacity_tile = self.tile
                return tiles
            del tiles
            self.capacity_limit //= 2

    def evaluate(self, batch, checkpoint):
        try:
            return super().evaluate(batch, checkpoint)
        except RuntimeError as exc:
            if self.capacity_limit is None or not is_cuda_allocation_error(exc):
                raise
            if self.tile <= 1024:
                if isinstance(exc, torch.OutOfMemoryError):
                    raise
                raise torch.OutOfMemoryError(str(exc)) from exc
        self.tile = self.capacity_limit = 1024
        return self.evaluate(batch, checkpoint)

    @torch.no_grad()
    def prepare_inference(self):
        """Pack the frozen CUDA distance heads and bilinear products."""
        if (
            not self.first.is_cuda
            or self.first.dtype != torch.float32
            or self.retain_messages
            or self.rep.training
            or torch.version.hip is not None
            or any(p.requires_grad for p in self.rep.parameters())
        ):
            return
        self.distance_weight = tangent_embedding(self.rep)
        if self.distance_weight is None:
            return
        self.heads = [
            tangent_head(inter, edge)
            for inter, edge in zip(
                self.rep.interactions, self.rep.edge_updates, strict=True
            )
        ]
        for product in self.rep.products:
            product.__class__ = FrozenBilinearProduct
            product.prepare()

    def layer(self, i, tile, vec, xs, xc):
        _, _, _, index, source, types, batch = tile
        if (
            self.distance_weight is None
            or not len(source)
            or vec.dtype != torch.float64
        ):
            return super().layer(i, tile, vec, xs, xc)
        rep, n = self.rep, len(batch)
        inter = rep.interactions[i]
        length = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        f, df = distance_inputs(rep, length, self.distance_weight)
        cut = rep.radial_basis.cutoff_fn(length).to(f.dtype)
        y = rep.o3_angular_basis(vec / length.clamp_min(1e-12)).to(f.dtype) * cut
        x = inter.linear_up(inter.norm1(xs)) if i else self.first[types[n:]]
        msg = tangent_message(
            inter,
            rep.edge_updates[i],
            x,
            y,
            length,
            f,
            df,
            cut,
            torch.stack((source, index[1])),
            n,
            types[index[0]],
            types[index[1]],
            save_hidden=torch.is_grad_enabled(),
            head=self.heads[i],
        )
        return self.update(i, tile, msg, xc), msg, None

    @staticmethod
    def prepare_radial(inter, update):
        last = inter.edge_info.mlp[-1]
        return LayerWeights(
            radial_parameters(inter, update), last.get_weight(), last.bias
        )

    def edge_operands(self, i, tile, vec, xs):
        _, _, _, index, _, types, batch = tile
        inter = self.rep.interactions[i]
        length = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        feat, attrs, cutoff = geometry(self.rep, vec, length, types, index)
        h, d = radial_heads(
            inter, self.radial[i].heads, feat, types[index[0]], types[index[1]]
        )
        x = inter.linear_up(inter.norm1(xs)) if i else self.first[types[len(batch) :]]
        return inter.rejector.reshape_in(x), attrs, h, d * cutoff

    def update(self, i, tile, y, xc):
        types, batch = tile[-2:]
        return update_node(
            self.rep.interactions[i],
            self.rep.products[i],
            y,
            xc,
            types[: len(batch)],
            batch,
        )
