"""Optional bounded-memory radial head for long OAM-L trajectories.

The joint two-head CUDA kernel is faster when its narrow edge state fits.
Here the convolution and density scatter are independent tiled EQX programs.
Neither saves an edge-by-128 or edge-by-64 hidden activation for its VJP.
Only the shared 64-channel input embedding and integer connectivity survive.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from equivariantx.ops.common.edge import evaluate
from equivariantx.utils.program import Program


class _Head(NamedTuple):
    weight: torch.Tensor
    bias: torch.Tensor
    target: torch.Tensor
    source: torch.Tensor
    it: torch.Tensor
    isrc: torch.Tensor
    gain: torch.Tensor
    shift: torch.Tensor
    eps: float

    mode = "full"

    def forward(self, inputs):
        y = torch.addmm(self.bias, inputs, self.weight)
        y = y + self.target[self.it] + self.source[self.isrc]
        return F.silu(
            F.layer_norm(y, (self.bias.numel(),), self.gain, self.shift, self.eps)
        )

    def to_edge_program(self, inputs, program, value):
        def shared(tensor):
            slot = len(inputs)
            inputs.append(tensor.reshape(1, -1))
            return program.input(slot, "shared", tensor.numel())

        k, n = self.weight.shape
        val = program.matmul(value, shared(self.weight), 1, k, n)
        val = program.binary("add", val, shared(self.bias))
        for table, rows in ((self.target, self.it), (self.source, self.isrc)):
            slot = len(inputs)
            inputs.extend((table, rows))
            element = program.input(slot, ("indexed", slot + 1, slot), n)
            val = program.binary("add", val, element)
        val = program.add("normalize", n, (val,), (self.eps, True))
        val = program.binary("mul", val, shared(self.gain))
        return program.unary("silu", program.binary("add", val, shared(self.shift)))


def streamed_heads(p, edge_feats, source_types, target_types, edge_index, nodes):
    """Return radial, projection, full-mode network and already-summed node density."""

    def head(section, eps):
        return _Head(
            p.edge[:, section].contiguous(),
            p.bias[section],
            p.target[:, section].contiguous(),
            p.source[:, section].contiguous(),
            target_types,
            source_types,
            p.gain[section],
            p.shift[section],
            eps,
        )

    info = head(slice(0, 128), p.info_eps)
    dens = head(slice(128, 192), p.density_eps)
    program, inputs = Program(), [edge_feats]
    val = dens.to_edge_program(inputs, program, program.input(0, "edge", 64))
    val = program.matmul(val, program.input(len(inputs), "shared", 64), 1, 64, 1)
    inputs.append(p.density.reshape(1, 64))
    val = program.binary("add", val, program.input(len(inputs), "shared", 1))
    inputs.append(p.density_bias.reshape(1, 1))
    root = program.unary("tanh", program.binary("mul", val, val))
    metadata = repr((tuple(program.nodes), ((root, 0, "target"),)))
    source, target = edge_index
    if edge_feats.is_cuda:
        (total,) = evaluate(metadata, inputs, source, target, nodes, inline=False)
    else:
        d = (
            torch.addmm(p.density_bias, dens.forward(edge_feats), p.density)
            .square()
            .tanh()
        )
        total = d.new_zeros((nodes, 1)).index_add(0, target, d)
    projection = torch.cat((p.projection, p.projection_bias[None]), 0)
    return edge_feats, projection, info, total
