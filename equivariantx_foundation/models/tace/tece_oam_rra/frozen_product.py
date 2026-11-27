"""Frozen output projections for the TECE-RRA bilinear product."""

import torch
from e3nn import o3

from equivariantx.o3 import ElementLinear
from tace.models.linear import has_lora
from tace.models.spherical.tece_oam_rra import BilinearMoEACE
from tace.ops.bilinear import BilinearContraction


def matrices(module):
    if module.use_matrix_weight:
        return list(module.weight)
    sizes = [a * b for a, b in module._path_shapes]
    return [
        w.unflatten(-1, shape)
        for w, shape in zip(module.weight.split(sizes, -1), module._path_shapes)
    ]


@torch.no_grad()
def fold_coefficients(grouped, shared, final, linear, types, scale):
    dest = final.weight[0] if final.use_matrix_weight else final.weight
    weights = dest.new_empty(len(types), linear.weight_numel)
    bias = dest.new_zeros(len(types), final.irreps_out.dim)
    offset = 0
    terms = zip(
        grouped.linear.instructions,
        shared.linear.instructions,
        final.linear.instructions,
        linear.instructions,
        matrices(grouped),
        matrices(shared),
        matrices(final),
        strict=True,
    )
    for gi, si, fi, wi, g, s, f in terms:
        if not (
            gi.i_in
            == gi.i_out
            == si.i_in
            == si.i_out
            == fi.i_in
            == fi.i_out
            == wi.i_in
            == wi.i_out
        ):
            raise ValueError("Expected one channel map per irrep.")
        ci, co = wi.path_shape
        experts, per_expert = grouped.num_experts, gi.path_shape[1]
        f = f.double() * (fi.path_weight * scale)
        s = (s.double() @ f) * (si.path_weight / wi.path_weight)
        experts_f = f.reshape(experts, per_expert, co)
        for begin in range(0, len(types), 4):
            ids = types[begin : begin + 4]
            indices = slice(begin, begin + len(ids))
            product = torch.einsum("zeij,ejk->zeik", g[ids].double(), experts_f)
            product = product.reshape(len(ids), ci, co).mul_(
                gi.path_weight / wi.path_weight
            )
            weights[indices, offset : offset + ci * co] = (product + s).flatten(1)
        if final.irreps_out[wi.i_out].ir.is_scalar():
            sb = shared.bias
            gb = grouped.bias
            if gb is not None or sb is not None:
                if len(shared._0e_slices) != 1 or len(grouped._bias_out_indices) != 1:
                    raise ValueError("Expected one scalar output block.")
                b = f.new_zeros(len(types), f.shape[0])
                if gb is not None:
                    b += gb[types].flatten(1)
                if sb is not None:
                    b += sb
                bias[:, final.irreps_out.slices()[wi.i_out]] = b @ f
        offset += ci * co
    return weights, bias


class FrozenBilinearProduct(torch.nn.Module):
    """Fold a frozen TECE-RRA product into per-element 64-channel output maps.

    Parameters
    ----------
    product : BilinearMoEACE
        Product in evaluation mode.
    backend : {"torch", "cuda"}
        Contraction backend.
    elements : sequence of int, optional
        Element IDs in local table order. Inputs then index this compact table.

    Examples
    --------
    Fold a small scalar-and-vector product for two of its three element types.
    The folded product accepts local type indices in the order of ``elements``:

    >>> irreps = o3.Irreps("2x0e + 2x1o")
    >>> product = BilinearMoEACE(
    ...     layer=0, num_layers=1, num_elements=3, Lmax=1, lmax=1,
    ...     num_channel=2, num_expert=2, num_channel_per_expert=1,
    ...     target_irreps=o3.Irreps("0e + 1o"), irreps_in=irreps,
    ...     correlation=[2], l1l2=None, nonlinear="bilinear_gate", use_shared_expert=True,
    ... ).eval().requires_grad_(False)
    >>> elements = torch.tensor([2, 0])
    >>> folded = FrozenBilinearProduct(product, backend="torch", elements=elements)
    >>> x, local = irreps.randn(3, -1), torch.tensor([0, 1, 0])
    >>> batch = torch.zeros_like(local)
    >>> expected = product(x, elements[local], None, batch)
    >>> torch.testing.assert_close(folded(x, local, None, batch), expected)
    """

    def __init__(self, product, *, backend="cuda", elements=None):
        super().__init__()
        if (
            product.training
            or product.correlation != 2
            or not product.use_shared_expert
        ):
            raise ValueError("Expected a frozen quadratic shared-expert product.")
        if any(p.requires_grad for p in product.parameters()) or any(
            has_lora(layer)
            for layer in (*product.coefs, *product.shared_coefs, product.linear)
        ):
            raise ValueError(
                "Expected frozen coefficient and output weights without adapters."
            )
        self.linear_up, self.nonlinearity = product.linear_up, product.nonlinearity
        self.irreps_in, self.irreps_out = product.irreps_in, product.irreps_out
        for key in (
            "use_bilinear_gate",
            "num_ace_gate_weights",
            "num_hidden_channel",
            "_double_hidden_slices",
            "_hidden_muls",
            "_hidden_ir_dims",
            "residual_scale",
        ):
            setattr(self, key, getattr(product, key))
        n = product.coefs[0].num_elements
        dest = next(product.parameters())
        types = (
            torch.arange(n, device=dest.device)
            if elements is None
            else torch.as_tensor(elements, device=dest.device, dtype=torch.long)
        )
        total_bias = dest.new_zeros(len(types), product.irreps_out.dim)
        for i, (grouped, shared) in enumerate(
            zip(product.coefs, product.shared_coefs, strict=True)
        ):
            direct = o3.Linear(
                grouped.irreps_in,
                product.irreps_out,
                internal_weights=False,
                shared_weights=False,
            )
            weight, bias = fold_coefficients(
                grouped, shared, product.linear, direct, types, product.scale
            )
            self.register_buffer(f"coefficient{i}", weight)
            total_bias += bias
            if i == 0:
                self.first = ElementLinear(direct, backend=backend)
            else:
                self.second = BilinearContraction(
                    product.aces[0].tp, direct, backend=backend
                )
        if product.linear.bias is not None:
            total_bias[:, product.linear._0e_slices[0]] += product.linear.bias.detach()
        self.register_buffer("bias", total_bias)

    def forward(self, node_feats, node_type, sc, batch):
        x, y, gates = BilinearMoEACE._linear_up_features(self, node_feats)
        out = self.first(x, self.coefficient0, node_type)
        out = (
            out
            + self.second(x, y, gates, self.coefficient1, node_type)
            + self.bias[node_type]
        )
        return torch.add(sc, out, alpha=self.residual_scale) if sc is not None else out
