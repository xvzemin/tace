"""Species coefficient maps for the released OAM-L architecture."""

import torch

from equivariantx.ops.common.linear import contract
from tace.models.linear import e3nnElementLinear
from tace.utils.env import acceleration_enabled


class _ScalarBias(torch.autograd.Function):
    """Add the element bias to the leading scalar channels."""

    generate_vmap_rule = True

    @staticmethod
    def forward(features, bias, types):
        width = bias.shape[1]
        scalars = features[:, :width] + bias.index_select(0, types)
        return (
            scalars
            if features.shape[1] == width
            else torch.cat((scalars, features[:, width:]), 1)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        features, bias, types = inputs
        ctx.shape, ctx.width = bias.shape, features.shape[1]
        ctx.save_for_backward(types)
        ctx.save_for_forward(types)

    @staticmethod
    def backward(ctx, gradient):
        (types,) = ctx.saved_tensors
        bias_grad = gradient.new_zeros(ctx.shape).index_add(
            0, types, gradient[:, : ctx.shape[1]]
        )
        return gradient, bias_grad, None

    @staticmethod
    def jvp(ctx, dfeatures, dbias, dtypes):
        if dbias is None:
            return dfeatures
        (types,) = ctx.saved_tensors
        if dfeatures is None:
            return torch.nn.functional.pad(
                dbias.index_select(0, types), (0, ctx.width - ctx.shape[1])
            )
        return _ScalarBias.forward(dfeatures, dbias, types)


def _map(metadata, output_width, x, table, types, rows=None, owners=None):
    if rows is not None and not torch._C._are_functorch_transforms_active():
        from .element_grouped import grouped

        return grouped(metadata, x, table, types, rows, owners)
    table = table.unsqueeze(1)  # EQX coefficient ABI: [species, experts=1, weights]
    empty = x.new_empty((0, output_width))
    if torch._C._are_functorch_transforms_active():
        return contract(metadata, 2, types, x, table, empty, indexed=True)
    # Bind the native contraction even for tiny symbolic tracing examples.
    # The generic inliner would otherwise emit per-atom expanded matrices.
    return torch.ops.eqx.element_linear(metadata, 2, types, x, table, empty)


class _FrozenMap(torch.autograd.Function):
    """Apply the transposed per-species matrix as a coalesced forward GEMV.

    Unlike a training VJP, the input-only transpose need not retain any primal
    feature. This matters in OAM-L: quadratic features have 16k columns.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, table, types, plan, rows, owners):
        forward, _, _, dout, _ = plan
        return _map(forward, dout, x, table, types, rows, owners)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, table, types, ctx.plan, rows, owners = inputs
        ctx.save_for_backward(table, types, rows, owners)
        ctx.save_for_forward(table, types)

    @staticmethod
    def backward(ctx, dy):
        table, types, rows, owners = ctx.saved_tensors
        _, reverse, din, _, blocks = ctx.plan
        # One shared [species, path, output, input] table, never one matrix per
        # atom. This constant transformation can be folded in a frozen export.
        transpose = torch.cat(
            [
                table[:, offset : offset + ci * co]
                .reshape(table.shape[0], ci, co)
                .transpose(1, 2)
                .reshape(table.shape[0], co * ci)
                for offset, ci, co in blocks
            ],
            dim=-1,
        )
        if torch.is_grad_enabled():
            rows = owners = None  # use the recursive EQX derivative rule
        return (
            _map(reverse, din, dy, transpose, types, rows, owners),
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def jvp(ctx, dx, dtable, dtypes, dplan, drows, downers):
        table, types = ctx.saved_tensors
        forward, _, _, dout, _ = ctx.plan
        return _map(forward, dout, dx, table, types)


class OAMLElementLinear(e3nnElementLinear):
    def prepare_plan(self):
        linear = self.eqx_linear
        self.scalar_prefix = (
            self.bias is not None
            and bool(self._0e_slices)
            and (self._0e_slices[-1].stop == self.bias.shape[-1])
        )
        rev = tuple(
            (sy, sw, sx, co, ci, dim, scale)
            for sx, sw, sy, ci, co, dim, scale in linear.paths
        )
        self.frozen_plan = (
            linear.metadata,
            repr(
                (
                    linear.output_dim,
                    linear.input_dim,
                    linear.weight_numel,
                    1,
                    rev,
                    linear.layout,
                )
            ),
            linear.input_dim,
            linear.output_dim,
            tuple((sw, ci, co) for _, sw, _, ci, co, _, _ in linear.paths),
        )

    def forward(self, x, node_type, groups=None):
        if (
            self.training
            and x.is_cuda
            and not self.use_matrix_weight
            and self.scalar_prefix
            and self.weight.requires_grad
            and acceleration_enabled("eqx", kernel="linear")
        ):
            value = self.eqx_linear(x, self.weight, node_type)
            return _ScalarBias.apply(value, self.bias, node_type)
        if not x.is_cuda or self.use_matrix_weight or self.weight.requires_grad:
            return super().forward(x, node_type)
        rows, owners = (None, None) if groups is None else groups
        out = _FrozenMap.apply(
            x, self.weight, node_type, self.frozen_plan, rows, owners
        )
        if self.bias is not None:
            out = (
                out
                + self.bias[node_type].index_select(-1, self._bias_index)
                * self._bias_mask
            )
        return out
