"""Single-rank TECE-RRA inference with complete receiver neighborhoods."""

import copy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ..tace_oam_l.lammps import replay_edge_capacity
from ..tace_oam_l.receiver import ReceiverTiles, TiledFoundationLammpsCalc
from .model import RRAInteraction, optimize_rra


class RRAReceiver(ReceiverTiles):
    """Evaluate alternating CGTP and rotary-attention layers in receiver tiles."""

    def __init__(self, model, tile, *, fold_products=False):
        self.model = model.readout_fn
        self.rep = self.model.representation
        self.tile = tile
        self.retain_messages = False
        self.replay_first_at = self.replay_prefix_at = None
        self.energy_domains = None
        self.coefficient_groups = {}
        self.first = self.rep.node_embedding.embedding.weight
        self.fold_products = fold_products
        self.product_elements = ()
        self.product_table = None
        self.fidelity = model.get_fidelity_idx()
        self.readout_widths = [0] * (len(self.rep.products) - 1) + [
            self.rep.irreps_outs[-1].dim
        ]

    def prepare_elements(self, types):
        """Cache folded products for the element IDs in the current structure."""
        if self.fold_products and types.is_cuda and types.numel():
            from .frozen_product import FrozenBilinearProduct

            active = types.unique()
            elements = tuple(active.tolist())
            if self.product_elements != elements:
                table = types.new_full((len(self.first),), -1)
                table[active] = torch.arange(len(elements), device=types.device)
                self.folded = [
                    FrozenBilinearProduct(product, elements=active)
                    for product in self.rep.products
                ]
                self.product_elements, self.product_table = elements, table

    def geometry(self, vec, types, idx, *, o3, o2):
        rep = self.rep
        length = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
        basis, cutoff = rep.radial_basis(length, types, idx, rep.atomic_numbers)
        basis, cutoff = basis.to(self.first.dtype), cutoff.to(self.first.dtype)
        radial = rep.edge_embedding(None, types, basis, idx, cutoff)
        attrs = rotation = inverse = None
        if o3:
            attrs = rep.o3_angular_basis(vec / length.clamp_min(1e-12)).to(basis.dtype)
        if o2:
            if vec.is_cuda:
                rotation = inverse = rep.o2_angular_basis.forward_packed(vec).to(
                    basis.dtype
                )
            else:
                rotation, inverse = (
                    v.to(basis.dtype) for v in rep.o2_angular_basis(vec)
                )
        return basis, cutoff, radial, attrs, rotation, inverse

    def tile_geometry(self, tile, vec, local):
        return self.geometry(
            vec,
            tile[-2],
            tile[3],
            o3=bool(len(vec)) and not local,
            o2=bool(len(vec)) and local,
        )

    def message(self, i, tile, vec, x, projected, local):
        _, _, _, idx, _, types, batch = tile
        n = len(batch)
        rep = self.rep
        inter = rep.interactions[i]
        basis, cutoff, radial, attrs, rotation, inverse = self.tile_geometry(
            tile, vec, local
        )
        feats = rep.edge_updates[i](x, types, radial, idx, cutoff)
        if not len(vec):
            m = x.new_zeros(n, inter._linear_down_irreps_in().dim)
        else:
            m = inter._compute_messages(
                projected, types, basis, feats, attrs, idx, cutoff, rotation, inverse
            )[:n]
        return inter.scatter_norm(inter.linear_down(m), feats, idx, cutoff, n)

    def layer(self, i, tile, vec, xs, xc):
        types, batch = tile[-2:]
        n = len(batch)
        rep = self.rep
        inter = rep.interactions[i]
        x = self.first[types] if i == 0 else torch.cat((xc, xs))
        projected = inter.linear_up(inter.norm1(x) if hasattr(inter, "norm1") else x)
        m = self.message(i, tile, vec, x, projected, isinstance(inter, RRAInteraction))
        m = inter.linear_nonlinearity(inter.nonlinearity(m))
        if hasattr(inter, "resnetBA"):
            m = torch.add(inter.resnetBA(xc), m, alpha=inter.residual_scale)
        sc = inter.resnetAB(m) if hasattr(inter, "resnetAB") else None
        if hasattr(inter, "norm2"):
            m = inter.norm2(m)
        if self.product_table is None:
            value = rep.products[i](m, types[:n], sc, batch)
        else:
            value = self.folded[i](m, self.product_table[types[:n]], sc, batch)
        if i == len(rep.products) - 1 and hasattr(rep, "final_norm"):
            value = rep.final_norm(value)
        return value, None, None


class RRACapacityLammpsCalc(TiledFoundationLammpsCalc):
    """Select compiled or receiver-tiled force evaluation for TECE-RRA."""

    model_label = "TECE-RRA"

    @staticmethod
    def build_receiver(model, *, tile, retain_messages):
        device = next(model.parameters()).device
        roomy = device.type == "cuda" and torch.cuda.mem_get_info(device)[0] >= (
            24 << 30
        )
        limit = 1024 if roomy else 512
        return RRAReceiver(model, min(tile, limit), fold_products=True)

    @staticmethod
    def indexed_workspace(edges):
        return edges * 500_000 + (3 << 30)

    def needs_tiles(self, data):
        return self.indexed_workspace(data.npairs) > self.workspace_budget


class StreamedRRALammpsCalc(RRACapacityLammpsCalc):
    """Use streamed attention before falling back to receiver tiles."""

    replay_atoms = 5_000
    replay_edges = 1 << 19
    bytes_per_edge = 96 << 10

    def _initialize_device(self, data):
        self.graph_max_atoms = self.replay_atoms
        self.graph_max_edges = self.replay_edges
        return super()._initialize_device(data)

    def needs_tiles(self, data):
        return (
            data.npairs * self.bytes_per_edge + data.nlocal * 96_000 + (3 << 30)
            > self.workspace_budget
        )


class ThroughputRRALammpsCalc(StreamedRRALammpsCalc):
    """Select indexed or streamed attention using the local edge workspace.

    Parameters
    ----------
    package_path, streamed : str or Path
        Indexed and streamed force packages for the same checkpoint.
    saved : str or Path, optional
        Streamed force package retaining local product intermediates.
    primal_edge_limit : int, optional
        Maximum number of edge rows with retained local product features.
    atomic_numbers : sequence of int
        Atomic numbers in model order.
    cutoff : float
        Interaction cutoff in angstroms.
    dtype : torch.dtype
        Model precision.
    """

    replay_atoms = 14_000
    replay_edges = 1 << 20
    bytes_per_edge = 64 << 10
    export_saved_primals = True
    saved = None
    primal_edge_limit = None
    graph_mode = "indexed"

    @staticmethod
    def cache_rows(device):
        from .force import EDGE_TILE

        total = torch.cuda.get_device_properties(device).total_memory
        return max(1, round(total / (3 << 29))) * EDGE_TILE

    @staticmethod
    def supports_saved_primals(updates):
        from .program import activation

        return all(
            layer._eqx_metadata is not None
            and layer.use_asymmetric_contraction
            and layer.num_head == 4
            and layer.linear_up.irreps_out.dim == 3520
            and layer.linear_coefs.irreps_out.dim == 3040
            and (activation(layer.scalar_act) or (None,))[0] == "silu"
            for layer in updates
        )

    def __init__(
        self,
        package_path,
        atomic_numbers,
        cutoff,
        dtype,
        *,
        streamed,
        saved=None,
        primal_edge_limit=None,
    ):
        super().__init__(package_path, atomic_numbers, cutoff, dtype)
        path = Path(streamed)
        self.streamed = path.name, path.read_bytes()
        self.primal_edge_limit = primal_edge_limit
        if saved is not None:
            path = Path(saved)
            self.saved = path.name, path.read_bytes()

    def saved_workspace(self, nodes, edges):
        from .force import EDGE_TILE

        padded = replay_edge_capacity(edges)
        capacity = (padded + EDGE_TILE - 1) // EDGE_TILE * EDGE_TILE
        wide = capacity * (164 << 10)
        limit = self.primal_edge_limit
        if limit is not None and capacity <= limit * 4 // 3:
            # Each of the four layers retains 3,520 and 3,040 FP32 components.
            wide = capacity * (44 << 10) + min(capacity, limit) * 4 * 4 * (3520 + 3040)
        return (3 << 30) + nodes * 96_000 + wide

    def graph_for(self, data):
        from .force import EDGE_TILE

        indexed = self.graph_mode == "indexed" and (
            self.indexed_workspace(data.npairs) <= self.workspace_budget
        )
        if indexed and (self.saved is None or data.npairs < 4 * EDGE_TILE):
            return "indexed"
        if (
            self.graph_mode != "streamed"
            and self.saved is not None
            and (
                self.saved_workspace(data.nlocal, data.npairs)
                < self.workspace_budget * 0.93
            )
        ):
            return "saved"
        return "indexed" if indexed else "streamed"

    def select_package(self, mode):
        self.graph_mode = mode
        if mode != "indexed":
            self.package_filename, self.package_bytes = (
                self.saved if mode == "saved" else self.streamed
            )
        print(f"EQX TECE-RRA force package: {self.package_filename}", flush=True)

    def _initialize_device(self, data):
        self.workspace_budget, _ = torch.cuda.mem_get_info(
            torch.as_tensor(data.elems).device
        )
        self.select_package(self.graph_for(data))
        return super()._initialize_device(data)

    def compute_forces(self, data):
        if self.initialized and self.receiver is None and self.graph_mode != "streamed":
            mode = self.graph_for(data)
            if mode != self.graph_mode:
                # Release the previous runner before loading a second force graph.
                torch.cuda.synchronize(self.device)
                self.force_graph = self.model = self.exported_forces = None
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                self.graph_mode = mode
                self._initialize_device(data)
        return super().compute_forces(data)


def export_lammps(model, output, *, capacity, aoti_package=None):
    """Export a frozen single-rank FP32 TECE-RRA force package."""
    from tace.models.compile import export_lammps_aotinductor

    from ..foundation import export_local_lammps
    from .frozen_product import FrozenBilinearProduct
    from .model import enable_frozen_expansion

    indexed = optimize_rra(model).eval().requires_grad_(False)
    memo = {id(p): p for p in model.parameters()}
    streamed = copy.deepcopy(model, memo.copy()).eval().requires_grad_(False)
    products = streamed.readout_fn.representation.products
    for i, product in enumerate(products):
        products[i] = FrozenBilinearProduct(product)
    options = dict(aoti_package=aoti_package, receiver_model=indexed)
    if capacity:
        return export_local_lammps(
            streamed, output, loader_class=StreamedRRALammpsCalc, **options
        )
    if (
        streamed.get_model_dtype() != torch.float32
        or next(streamed.parameters()).device.type != "cuda"
    ):
        raise ValueError("Single-rank foundation export requires FP32 on CUDA.")
    native = copy.deepcopy(indexed, memo)
    native.readout_fn.representation.products = products
    enable_frozen_expansion(native)
    with TemporaryDirectory() as directory:
        secondary = export_lammps_aotinductor(
            streamed, Path(directory) / "rra-streamed.pt2", local_owners=True
        )
        saved = limit = None
        updates = [
            layer.rejector
            for layer in streamed.readout_fn.representation.interactions[1::2]
        ]
        if (
            ThroughputRRALammpsCalc.export_saved_primals
            and ThroughputRRALammpsCalc.supports_saved_primals(updates)
        ):
            streamed.readout_fn.representation.o2_angular_basis.method = "generator"
            limit = ThroughputRRALammpsCalc.cache_rows(
                next(streamed.parameters()).device
            )
            for layer in updates:
                layer.save_force_primals = True
                layer.primal_edge_limit = limit
            saved = export_lammps_aotinductor(
                streamed, Path(directory) / "rra-primal.pt2", local_owners=True
            )
        return export_local_lammps(
            native,
            output,
            loader_class=partial(
                ThroughputRRALammpsCalc,
                streamed=secondary,
                saved=saved,
                primal_edge_limit=limit,
            ),
            **options,
        )
