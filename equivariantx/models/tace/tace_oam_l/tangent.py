"""Frozen radial-tangent force evaluation for the TACE-OAM-L checkpoint."""

from pathlib import Path

import torch

from equivariantx.ops.common.cuda.o3.convolution import contract_direct
from equivariantx.ops.common.cuda.pointwise import launch, scalar_type
from equivariantx.utils.metadata import compact_metadata

from .angular import edge_attributes
from .convolution import _affine
from .element_grouped import grouping
from .model import _OAMLRepresentation, optimize_oam_l
from .projection import edge_limit, project
from .radial import RadialParameters, _kernel, prepare_radial, radial_heads
from .tangent_kernels import coordinate_adjoint, encode_tangent, heads

HEAD_TILE = 65_536


@torch.library.custom_op("eqx_oam_l_tangent::embedding", mutates_args=())
def embedding(
    r: torch.Tensor,
    roots: torch.Tensor,
    prefactor: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return encode_tangent(r, roots, prefactor, weight, scale)


@embedding.register_fake
def fake_embedding(r, roots, prefactor, weight, scale):
    size = (r.shape[0], weight.shape[1])
    return weight.new_empty(size), weight.new_empty(size)


def full_message(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    feats: torch.Tensor,
    fdot: torch.Tensor,
    st: torch.Tensor,
    tt: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    shape: torch.Tensor,
    packed: list[torch.Tensor],
    eps1: float,
    eps2: float,
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = RadialParameters(*packed, eps1, eps2)
    h, d = radial_heads(p, feats, st, tt)
    return _affine(x, y, h, p.projection, p.projection_bias, src, dst, shape, meta), d


message = torch.library.custom_op(
    "eqx_oam_l_tangent::message", full_message, mutates_args=()
)


@torch.library.custom_op(
    "eqx_oam_l_tangent::stream_message",
    mutates_args=(),
    device_types="cuda",
    schema=torch.library.infer_schema(full_message, mutates_args=()),
)
def stream_message(
    x, y, r, feats, fdot, st, tt, src, dst, shape, packed, eps1, eps2, meta
):
    if len(feats) < 16 * HEAD_TILE:
        return full_message(
            x, y, r, feats, fdot, st, tt, src, dst, shape, packed, eps1, eps2, meta
        )

    p = RadialParameters(*packed, eps1, eps2)
    z, density = x.new_zeros(shape.shape), x.new_empty(len(feats), 1)
    pre = x.new_empty(HEAD_TILE, p.edge.size(1))
    hidden = x.new_empty(HEAD_TILE, p.projection.size(0))
    weights = x.new_empty(HEAD_TILE, p.projection.size(1))
    code = _kernel(scalar_type(st.dtype), eps1, eps2, False, hidden.size(1))
    for start in range(0, len(feats), HEAD_TILE):
        sl = slice(start, start + HEAD_TILE)
        n = len(feats[sl])
        a, h, w = pre[:n], hidden[:n], weights[:n]
        torch.mm(feats[sl], p.edge, out=a)
        launch(
            code,
            [a, st[sl], tt[sl], *p[1:8], h, density[sl]],
            (n,),
            n * 32,
            threads=128,
        )
        torch.addmm(p.projection_bias, h, p.projection, out=w)
        contract_direct(
            meta, src[sl], dst[sl], [((4,), (x, w, w[:0], y[sl], z), (z,), False)]
        )
    return z, density


@message.register_fake
def fake_message(
    x, y, r, feats, fdot, st, tt, src, dst, shape, packed, eps1, eps2, meta
):
    return x.new_empty(shape.shape), feats.new_empty((feats.shape[0], 1))


@torch.library.custom_op("eqx_oam_l_tangent::vjp", mutates_args=())
def vjp(
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    feats: torch.Tensor,
    fdot: torch.Tensor,
    st: torch.Tensor,
    tt: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    g: torch.Tensor,
    gd: torch.Tensor,
    packed: list[torch.Tensor],
    eps1: float,
    eps2: float,
    meta: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = RadialParameters(*packed, eps1, eps2)
    dx, dy, dr = torch.zeros_like(x), torch.zeros_like(y), feats.new_zeros((len(y), 1))
    g = g.contiguous()
    # The same tile bounds apply to the path values and their scalar-distance derivative.
    tile = min(32768, len(y))
    w, b = p.projection, p.projection_bias
    weights, tangent = (feats.new_empty(tile, w.shape[1]) for _ in range(2))
    for start in range(0, len(y), 32768):
        sl = slice(start, start + tile)
        h, _, hd, dd = heads(p, feats[sl], fdot[sl], st[sl], tt[sl])
        rows, deriv = weights[: len(h)], tangent[: len(h)]
        torch.addmm(b, h, w, out=rows)
        torch.mm(hd, w, out=deriv)
        torch.mul(gd[sl], dd, out=dr[sl])
        coordinate_adjoint(
            meta, src[sl], dst[sl], x, y[sl], rows, g, dx, dy[sl], dr[sl], deriv
        )
    return dx, dy, dr.to(dtype=r.dtype)


@vjp.register_fake
def fake_vjp(x, y, r, feats, fdot, st, tt, src, dst, g, gd, packed, eps1, eps2, meta):
    return torch.empty_like(x), torch.empty_like(y), torch.empty_like(r)


def setup(ctx, inputs, output):
    x, y, r, feats, fdot, st, tt, src, dst, _, packed, ctx.eps1, ctx.eps2, ctx.meta = (
        inputs
    )
    ctx.save_for_backward(x, y, r, feats, fdot, st, tt, src, dst, *packed)


def reverse(ctx, g, gd):
    x, y, r, feats, fdot, st, tt, src, dst, *packed = ctx.saved_tensors
    dx, dy, dr = vjp(
        x,
        y,
        r,
        feats,
        fdot,
        st,
        tt,
        src,
        dst,
        g,
        gd,
        packed,
        ctx.eps1,
        ctx.eps2,
        ctx.meta,
    )
    return (
        dx,
        dy,
        dr,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        [None] * len(packed),
        None,
        None,
        None,
    )


message.register_autograd(reverse, setup_context=setup)
stream_message.register_fake(fake_message)
stream_message.register_autograd(reverse, setup_context=setup)


class RadialTangentRepresentation(_OAMLRepresentation):
    saved_edge_limit = 0
    stream_heads = False

    def forward(self, data, graph):
        types, index = graph.node_type, data["edge_index"]
        local = graph.lmp_natoms[0] if graph.lmp else None
        local_types = types[:local]
        groups = grouping(local_types, self.num_elements)
        src, dst = index
        st, tt = types[src].contiguous(), types[dst].contiguous()
        length = graph.edge_length
        if length.dtype != torch.float64:
            raise ValueError("Radial-tangent OAM-L requires FP64 edge geometry.")
        basis = self.radial_basis.radial_fn
        feats, fdot = embedding(
            length.detach(),
            basis.bessel_weights,
            basis.prefactor,
            self.frozen_embedding,
            self.edge_embedding.act1.acts[0].cst,
        )
        cutoff = self.radial_basis.cutoff_fn(length)
        attrs = edge_attributes(graph.edge_vector, length, cutoff, self.model_dtype)
        table = self.interactions[0].linear_up(self.node_embedding.embedding.weight)
        x = torch.nn.functional.embedding(types, table)
        descriptors = []
        for i, (inter, update, product) in enumerate(
            zip(self.interactions, self.edge_updates, self.products)
        ):
            residual = None
            if i:
                residual = inter.resnetBB(x, local_types, groups)
                x = inter.linear_up(x)
                x = inter.handle_lammps(x, graph.lmp_data, graph.lmp_natoms, i)
            p = prepare_radial(inter, update)
            conv = inter.rejector
            inp = conv.reshape_in(x)
            dummy = x.new_empty(1).expand(local_types.shape[0], conv.eqx_tp.output_dim)
            operands = (
                inp.contiguous(),
                attrs.contiguous(),
                length,
                feats,
                fdot,
                st,
                tt,
                src.contiguous(),
                dst.contiguous(),
                dummy,
                [q.contiguous() for q in p[:10]],
                p.info_eps,
                p.density_eps,
                compact_metadata(conv.eqx_tp.kernel_metadata),
            )
            if i and self.saved_edge_limit:
                z, d = project(*operands, self.saved_edge_limit)[:2]
            else:
                z, d = (stream_message if self.stream_heads else message)(*operands)
            norm = inter.scatter_norm
            total = d.new_zeros((local_types.shape[0], 1)).index_add(0, dst, d)
            x = inter.linear_down(conv.reshape_out(z)) / (
                total[:local] * norm.beta + norm.alpha
            )
            x = inter.linear_nonlinearity(inter.nonlinearity(x))
            x = product(x, local_types, residual, data["batch"], groups)
            descriptors.append(x)
        return {
            "descriptors": descriptors,
            "uie_feats": None,
            "noise_mask_tensor": None,
            "dens_batch_mask_tensor": None,
            "magnetic_radial_basis": None,
        }


def export_tangent_lammps(model, output, *, aoti_package=None) -> Path:
    """Export the frozen single-rank OAM-L energy and pair-force graph.

    With the CUDA model loaded, ``export_tangent_lammps(model, "oam_l.pt")``
    writes the loader to ``oam_l.pt`` and the compiled graph to ``oam_l.pt2``.

    Parameters
    ----------
    model : torch.nn.Module
        FP32 CUDA TACE-OAM-L model.
    output : str or Path
        Destination of the LAMMPS ML-IAP loader.
    aoti_package : str or Path, optional
        Compiled graph destination. Defaults to the output with suffix `.pt2`.
    """
    from tace.models.compile import export_lammps_aotinductor

    from .lammps import RadialTangentLammpsCalc

    output = Path(output)
    aoti_package = (
        Path(aoti_package) if aoti_package is not None else output.with_suffix(".pt2")
    )
    if aoti_package.resolve() == output.resolve():
        raise ValueError("The ML-IAP loader and AOTI package require distinct paths.")
    model = frozen_tangent_model(model, save_projection=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    aoti_package.parent.mkdir(parents=True, exist_ok=True)
    package = export_lammps_aotinductor(model, aoti_package, local_owners=True)
    calc = RadialTangentLammpsCalc(
        package, model.get_atomic_numbers(), model.get_cutoff(), model.get_model_dtype()
    )
    calc.graph_max_atoms = 1024
    torch.save(calc, output, pickle_protocol=5)
    return output


def radial_embedding(rep):
    """Return the frozen eight-mode embedding matrix, or None for other bases.

    Examples
    --------
    The tangent kernel uses a constant matrix in place of the bias-free
    projection. The matrix reproduces the projection before activation:

    >>> from tace.models.radial import RadialBasis
    >>> from tace.models.spherical.edge import NonLinearEdgeEmbedding
    >>> rep = torch.nn.Module()
    >>> rep.radial_basis = RadialBasis(num_basis=8, apply_cutoff=False)
    >>> rep.edge_embedding = NonLinearEdgeEmbedding(
    ...     num_elements=2, num_radial_basis=8, num_channel=64,
    ... ).float()
    >>> weight = radial_embedding(rep)
    >>> weight.shape, weight.requires_grad
    (torch.Size([8, 64]), False)
    >>> modes = torch.randn(3, 8, dtype=torch.float32)
    >>> torch.testing.assert_close(modes @ weight, rep.edge_embedding.radial_proj(modes))
    """
    basis, edge = rep.radial_basis, rep.edge_embedding
    if (
        basis.apply_cutoff
        or basis.use_distance_transform
        or type(basis.radial_fn).__name__ != "j0SphericalBesselBasis"
        or basis.num_basis != 8
        or edge.radial_proj.bias is not None
        or basis.radial_fn.bessel_weights.dtype != torch.float64
        or basis.radial_fn.prefactor.dtype != torch.float64
    ):
        return None
    eye = torch.eye(8, device=basis.radial_fn.prefactor.device, dtype=torch.float32)
    return edge.radial_proj(eye).detach().contiguous()


def receiver_pullback(rep, weight, p, meta, x, node_grad, vec, idx, src, types, g, gd):
    """Contract a receiver tile's angular and radial derivatives with its message adjoint."""
    basis = rep.radial_basis
    q = vec.detach().requires_grad_()
    r = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    attrs = edge_attributes(q, r, basis.cutoff_fn(r), rep.model_dtype)
    with torch.no_grad():
        dx = torch.zeros_like(x) if node_grad else x.new_empty(0)
        da = torch.zeros_like(attrs)
        dr = torch.empty_like(r, dtype=x.dtype)
        width = p.projection.size(1)
        size = min(len(src), 32768)
        values, tangents = (x.new_empty(size, width) for _ in range(2))
        for start in range(0, len(src), size):
            sl = slice(start, start + size)
            target = idx[1, sl]
            f, fdot = encode_tangent(
                r[sl],
                basis.radial_fn.bessel_weights,
                basis.radial_fn.prefactor,
                weight,
                rep.edge_embedding.act1.acts[0].cst,
            )
            h, _, hd, dd = heads(p, f, fdot, types[idx[0, sl]], types[target])
            rows, dots = values[: len(target)], tangents[: len(target)]
            torch.addmm(p.projection_bias, h, p.projection, out=rows)
            torch.mm(hd, p.projection, out=dots)
            torch.mul(gd[target], dd, out=dr[sl])
            coordinate_adjoint(
                meta, src[sl], target, x, attrs[sl], rows, g, dx, da[sl], dr[sl], dots
            )
    (dv,) = torch.autograd.grad((attrs, r), q, (da, dr.to(r.dtype)))
    return dx, dv


def frozen_tangent_model(model, *, save_projection=False):
    """Prepare the frozen OAM-L energy and first-derivative graph.

    With the CUDA model loaded, ``frozen_tangent_model(model)`` prepares the
    streaming force graph; pass ``save_projection=True`` to reuse projections.

    Parameters
    ----------
    model : torch.nn.Module
        FP32 CUDA TACE-OAM-L model.
    save_projection : bool, optional
        Retain a bounded prefix of the deep radial projections for their adjoint.
    """
    if (
        model.get_model_dtype() != torch.float32
        or next(model.parameters()).device.type != "cuda"
    ):
        raise ValueError("Radial-tangent OAM-L requires a frozen FP32 CUDA export.")
    model = optimize_oam_l(
        model.eval().requires_grad_(False),
        fused_geometry=True,
        fused_projection=True,
        grouped_coefficients=True,
    )
    rep = model.readout_fn.representation
    embedding_weight = radial_embedding(rep)
    if embedding_weight is None:
        raise ValueError(
            "Radial-tangent OAM-L requires its bias-free eight-mode j0 embedding."
        )
    rep.__class__ = RadialTangentRepresentation
    rep.stream_heads = not save_projection
    rep.saved_edge_limit = (
        edge_limit(rep, next(model.parameters()).device) if save_projection else 0
    )
    from .product_kernels import square_metadata

    for product in rep.products:
        if not product.scalar_norms:
            product.native_square = square_metadata(product)
    rep.register_buffer("frozen_embedding", embedding_weight, persistent=False)
    return model
