"""CUDA graph replay for atomic energies and pair forces."""

import torch


def replay_edge_capacity(edges):
    """Return the number of edge rows allocated for a force-graph capture."""
    return ((edges + max(8, edges // 50) + 63) // 64) * 64


class ForceReplay:
    """Replay an AOTI local-owner force graph within a fixed edge capacity."""

    def __init__(self, model, batch, n, e, device):
        from equivariantx.ops.common.cuda.graph import retain_constants

        self.constants = {}
        self.n = n
        self.capacity = replay_edge_capacity(e)
        self.vectors = torch.empty(
            (self.capacity, 3), device=device, dtype=torch.float64
        )
        self.indices = torch.empty((2, self.capacity), device=device, dtype=torch.long)
        self.types = torch.zeros(n + 1, device=device, dtype=torch.long)
        static = dict(
            edge_vector=self.vectors,
            node_type=self.types,
            edge_index=self.indices,
            batch=self.types.new_zeros(n + 1),
            ptr=self.types.new_tensor([0, n + 1]),
            fidelity_idx=self.types.new_tensor([model.fidelity_idx]),
        )
        self.copy(batch)
        self.args = tuple(static[key] for key in model.input_keys)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), retain_constants(self.constants):
            model.compiled_model(*self.args)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with (
            retain_constants(self.constants),
            torch.cuda.graph(
                self.graph, stream=stream, capture_error_mode="thread_local"
            ),
        ):
            self.energy, self.energies, self.forces = model.compiled_model(*self.args)

    def copy(self, batch):
        e = batch["edge_vector"].shape[0]
        self.vectors[:e].copy_(batch["edge_vector"].detach())
        self.indices[:, :e].copy_(batch["edge_index"])
        self.types[: self.n].copy_(batch["node_type"])
        # Keep dummy edges entirely in a disconnected, unobserved component.
        self.vectors[e:].zero_()
        self.vectors[e:, 0].fill_(2.0)
        self.indices[:, e:].fill_(self.n)

    def __call__(self, batch):
        self.copy(batch)
        self.graph.replay()
        return self.energies[: self.n], self.forces[: batch["edge_vector"].shape[0]]
