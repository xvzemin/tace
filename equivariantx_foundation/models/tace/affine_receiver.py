"""Shared affine message and its VJP for compatible frozen OAM adapters."""

import torch

from .receiver import TACEReceiver
from .tace_oam_l.convolution import MessageTranspose, bias_convolution


class AffineOAMReceiver(TACEReceiver):
    """OAM affine convolution, normalized-message tape and its explicit VJP.

    The 7M receiver shares this ABI for ordinary (non-distance-tangent) tiles,
    and supplies its own operands, radial parameters, and node update.
    """

    def __init__(self, model, tile, retain_messages=False):
        self.model = model.readout_fn
        self.rep = self.model.representation
        self.tile = tile
        self.retain_messages = retain_messages
        self.replay_first_at = None
        self.replay_prefix_at = None
        self.message_transposes = (
            [
                MessageTranspose(i.linear_down, i.rejector.irreps_out)
                for i in self.rep.interactions
            ]
            if retain_messages
            else []
        )
        self.radial = [
            self.prepare_radial(inter, update)
            for inter, update in zip(self.rep.interactions, self.rep.edge_updates)
        ]
        self.first = self.rep.interactions[0].linear_up(
            self.rep.node_embedding.embedding.weight
        )
        self.fidelity = model.get_fidelity_idx()
        self.readout_widths = []
        for i, irreps in enumerate(self.rep.irreps_outs):
            if not self.model.use_alllayer and i < len(self.radial) - 1:
                self.readout_widths.append(0)
                continue
            readout = self.model.energy_readouts[i if self.model.use_alllayer else -1]
            first = (readout.linear2 if readout.last_layer else readout.linear1)[0]
            # Keep every input block consumed by the first readout linear.
            used = {ir for _, ir in first.irreps_out}
            self.readout_widths.append(
                max(
                    (
                        s.stop
                        for (_, ir), s in zip(irreps, irreps.slices())
                        if ir in used
                    ),
                    default=0,
                )
            )

    def layer(self, i, tile, vec, xs, xc):
        local, nodes, edges, idx, s, z, batch = tile
        inter, params, n = self.rep.interactions[i], self.radial[i], len(batch)
        conv = inter.rejector
        if len(s):
            x, attr, h, density = self.edge_operands(i, tile, vec, xs)
            message = bias_convolution(
                conv.eqx_tp,
                x,
                attr,
                h,
                params.projection,
                params.projection_bias,
                torch.stack((s, idx[1])),
                n,
            )
            total = density.new_zeros((n, 1)).index_add_(0, idx[1], density)
        else:
            message = self.first.new_zeros(n, conv.eqx_tp.output_dim)
            total = self.first.new_zeros(n, 1)
        norm = inter.scatter_norm
        denom = norm.alpha + norm.beta * total
        y = inter.linear_down(conv.reshape_out(message)) / denom
        return self.update(i, tile, y, xc), y, denom

    def adjoint(self, i, tile, vec, xs, xc, incoming, y, denom):
        from equivariantx.utils.metadata import compact_metadata

        inter = self.rep.interactions[i]
        conv = inter.rejector
        s = tile[4]
        z = y.detach().requires_grad_()
        output = self.update(i, tile, z, xc)
        grad = torch.autograd.grad(output, (z, xc) if i else (z,), incoming)
        dz = grad[0]
        if not len(s):
            return (None, None, grad[1]) if i else (None,)
        message_grad = self.message_transposes[i](dz / denom)
        density_grad = -(dz * y).sum(-1, keepdim=True) * inter.scatter_norm.beta / denom
        metadata = compact_metadata(conv.eqx_tp.kernel_metadata)
        x, dx, dv = self.edge_vjp(
            i, tile, vec, xs, message_grad, density_grad, metadata
        )
        if i:
            (dsource,) = torch.autograd.grad(x, xs, dx)
            return dv, dsource, grad[1]
        return (dv,)

    def edge_vjp(self, i, tile, vec, xs, message_grad, density_grad, metadata):
        idx, s = tile[3:5]
        params = self.radial[i]
        x, attrs, h, density = self.edge_operands(i, tile, vec, xs)
        dx, da, dh = torch.ops.eqx_oam_l_conv.affine_vjp(
            x.contiguous(),
            attrs.contiguous(),
            h.contiguous(),
            params.projection.contiguous(),
            params.projection_bias.contiguous(),
            s,
            idx[1],
            message_grad.contiguous(),
            metadata,
            bool(i),
            True,
            True,
        )
        (dv,) = torch.autograd.grad(
            (attrs, h, density), vec, (da, dh, density_grad[idx[1]])
        )
        return x, dx, dv
