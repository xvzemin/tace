"""Frozen product and radial-tangent representations for TACE-OAM-7M."""

import torch

from equivariantx.utils.metadata import compact_metadata
from tace.models.spherical.tece_oam_rra import BilinearMoEACE

from .model import SevenRepresentation
from .product import angular, angular_plan, matrix_blocks
from .radial_batch import batch_radial
from .reciprocal import reciprocal_heads, reciprocal_projection
from .tangent import distance_inputs, encoded, pack, projected, tangent_message


class FrozenBilinearProduct(BilinearMoEACE):
    """Apply the gated 7M channel-wise product and its frozen output matrices."""

    @torch.no_grad()
    def prepare(self):
        a, b = self.coefs
        left, right, output = map(matrix_blocks, (a, b, self.linear))
        bias = a.bias.double() + b.bias.double()
        self.blocks = []
        self.angular_plan = angular_plan(self.aces[0].tp)
        self.angular_dim = self.aces[0].tp.irreps_out.dim
        for i, (co, ir) in enumerate(self.irreps_out):
            j, w = output[i]
            ia, u = left[j]
            ib, v = right[j]
            dtype = self.linear.bias.dtype
            self.register_buffer(f"matrix{i}", (torch.cat((u, v)) @ w).to(dtype))
            if ir.is_scalar():
                self.register_buffer(
                    "merged_bias", (bias @ w + self.linear.bias).to(dtype)
                )
            self.blocks.append(
                (
                    a.irreps_in.slices()[ia].start,
                    b.irreps_in.slices()[ib].start,
                    u.size(0),
                    v.size(0),
                    co,
                    ir.dim,
                    ir.is_scalar(),
                )
            )

    def forward(self, node_feats, node_type, sc, batch):
        x, base, gates = self._linear_up_features(node_feats)
        y = angular(x, base, gates, self.angular_plan, self.angular_dim)
        n = x.shape[0]
        blocks = []
        for i, (s, t, a, b, out, d, scalar) in enumerate(self.blocks):
            inputs = (
                torch.cat(
                    (
                        x.narrow(1, s, a * d).reshape(n, a, d),
                        y.narrow(1, t, b * d).reshape(n, b, d),
                    ),
                    1,
                )
                .transpose(1, 2)
                .reshape(n * d, a + b)
            )
            w = getattr(self, f"matrix{i}")
            z = torch.addmm(self.merged_bias, inputs, w) if scalar else inputs @ w
            blocks.append(z.reshape(n, d, out).transpose(1, 2).reshape(n, out * d))
        result = torch.cat(blocks, 1)
        return (
            torch.add(sc, result, alpha=self.residual_scale)
            if sc is not None
            else result
        )


class SevenDistanceTangent(SevenRepresentation):
    """Evaluate frozen 7M interactions and their first coordinate derivatives."""

    save_hidden = False

    def edge_geometry(self, vector, length, types, index):
        if length.dtype != torch.float64 or self.model_dtype != torch.float32:
            raise ValueError(
                "The frozen 7M distance tangent requires FP64 geometry and FP32 weights."
            )
        feat, dt = distance_inputs(self, length)
        cutoff = self.radial_basis.cutoff_fn(length).to(self.model_dtype)
        attrs = self.o3_angular_basis(vector / length.clamp_min(1e-12)).to(
            self.model_dtype
        )
        return attrs * cutoff, length, feat, dt, cutoff

    def edge_message(self, inter, edge, features, geometry, index, n, src, dst):
        return tangent_message(
            inter, edge, features, *geometry, index, n, src, dst, self.save_hidden
        )


class BatchedSeven(SevenDistanceTangent):
    """Evaluate all five frozen radial heads together before message passing."""

    save_projection = False
    reciprocal = False

    def prepare(self):
        for layer in self.products:
            layer.__class__ = FrozenBilinearProduct
            layer.prepare()

    def edge_geometry(self, vector, length, types, index):
        y, r, f, dt, cut = super().edge_geometry(vector, length, types, index)
        pars = [pack(i, e) for i, e in zip(self.interactions, self.edge_updates)]
        p = [
            torch.cat([v[0] for v in pars], 1).contiguous(),
            *(torch.stack([v[i] for v in pars]).contiguous() for i in range(1, 12)),
        ]
        layer = self.interactions[0].edge_info.mlp
        if self.reciprocal:
            h, d, dh, dd, inverse = reciprocal_heads(
                f, dt, index[0], index[1], types, p, layer[1].eps, layer[4].eps
            )
            return y, r, h, d, dh, dd, cut, inverse
        h, d, dh, dd = batch_radial(
            f, dt, types[index[0]], types[index[1]], p, layer[1].eps, layer[4].eps
        )
        return y, r, h, d, dh, dd, cut

    def edge_message(self, inter, edge, features, geometry, index, n, src, dst):
        y, r, h, d, dh, dd, cut = geometry[:7]
        i = inter.layer
        d = d[i] + (r - r.detach()).to(d.dtype) * dd[i]
        c = inter.rejector
        x = c.reshape_in(features).contiguous()
        w = inter.edge_info.mlp[-1]
        args = (
            x,
            y.contiguous(),
            r,
            h[i],
            dh[i],
            index[0].contiguous(),
            index[1].contiguous(),
            x.new_empty(1).expand(n, c.eqx_tp.output_dim),
            w.get_weight().contiguous(),
            w.bias.contiguous(),
            compact_metadata(c.eqx_tp.kernel_metadata),
        )
        if self.reciprocal:
            z = reciprocal_projection(*args[:-1], geometry[7], args[-1])[0]
        else:
            z = projected(*args)[0] if self.save_projection else encoded(*args)
        norm = inter.scatter_norm
        denom = d.new_zeros((n, 1)).index_add_(0, index[1], d * cut)
        return inter.linear_down(c.reshape_out(z)) / (norm.alpha + norm.beta * denom)
