"""Adapt the TACE energy readout to a graph of receiver rows and their halo."""

import torch

from equivariantx_foundation.execution.receiver import LayerwiseReceiver
from tace.models.lammps import Graph


class TACEReceiver(LayerwiseReceiver):
    @property
    def state_template(self):
        return self.first

    @property
    def state_widths(self):
        return tuple(ir.dim for ir in self.rep.irreps_outs)

    def readout(self, tile, vec, descriptors, total_atoms):
        local, nodes, edges, idx, s, z, batch = tile
        n = len(batch)
        graph = Graph(
            True,
            None,
            (n, len(nodes)),
            1,
            None,
            vec.new_zeros(n, 3),
            vec,
            torch.linalg.vector_norm(vec, dim=-1, keepdim=True),
            vec.new_zeros(1, 3, 3),
            torch.full_like(batch, self.fidelity),
            torch.arange(n, device=z.device),
            z,
        )
        data = dict(batch=batch, ptr=batch.new_tensor([0, total_atoms]), edge_index=idx)
        out = self.model.readout_fn(data, graph, {"descriptors": descriptors}, None)
        return out["node_energy"]
