"""Two-head, element-conditioned radial network of TACE-OAM-L.

The first affine projections in the 128-wide convolution and 64-wide
density heads have the same inputs. Project the shared nonlinear radial
embedding once, and project each endpoint table once per element instead
of once per edge. The wide final convolution projection remains separate.
All tables are functions of live parameters, not of coordinates or graph
topology. The module does not retain them between calls.
"""

from functools import lru_cache
from typing import NamedTuple

import torch
from torch.nn import functional as F


class RadialParameters(NamedTuple):
    edge: torch.Tensor
    target: torch.Tensor
    source: torch.Tensor
    bias: torch.Tensor
    gain: torch.Tensor
    shift: torch.Tensor
    density: torch.Tensor
    density_bias: torch.Tensor
    projection: torch.Tensor
    projection_bias: torch.Tensor
    info_eps: float
    density_eps: float


def validate_radial(inter):
    """Check the biased affine, LayerNorm and SiLU radial heads."""
    from tace.models.linear import mlpLinear

    info = inter.edge_info.mlp
    density = inter.scatter_norm.edge_density.mlp
    if not all(
        len(head) == 4
        and type(head[0]) is mlpLinear
        and type(head[-1]) is mlpLinear
        and type(head[1]) is torch.nn.LayerNorm
        and head[1].weight is not None
        and head[1].bias is not None
        and type(head[2]) is torch.nn.SiLU
        and head[0].in_dim == 192
        and head[0].out_dim == width
        and head[0].bias is not None
        and head[-1].bias is not None
        for head, width in ((info, 128), (density, 64))
    ):
        raise ValueError("TACE-OAM-L requires 128/64-wide biased LayerNorm-SiLU heads")


def prepare_radial(inter, update):
    """Fold both live affine heads and their endpoint embedding tables."""
    info = inter.edge_info.mlp
    density = inter.scatter_norm.edge_density.mlp
    merged = torch.cat((info[0].get_weight(), density[0].get_weight()), 1)
    edge, target, source = merged.split(64, dim=0)
    return RadialParameters(
        edge,
        update.target_embedding.weight @ target,
        update.source_embedding.weight @ source,
        torch.cat((info[0].bias, density[0].bias)),
        torch.cat((info[1].weight, density[1].weight)),
        torch.cat((info[1].bias, density[1].bias)),
        density[-1].get_weight(),
        density[-1].bias,
        info[-1].get_weight(),
        info[-1].bias,
        info[1].eps,
        density[1].eps,
    )


def _torch_heads(p, feat, source, target):
    y = torch.addmm(p.bias, feat, p.edge)
    y = y + p.target.index_select(0, target) + p.source.index_select(0, source)
    a, b = y.split((128, 64), -1)
    info = F.silu(F.layer_norm(a, (128,), p.gain[:128], p.shift[:128], p.info_eps))
    dens = F.silu(F.layer_norm(b, (64,), p.gain[128:], p.shift[128:], p.density_eps))
    dens = torch.addmm(p.density_bias, dens, p.density)
    return info, dens.square().tanh()


_TILE = 32768  # 24 MiB of temporary edge-affine activations, independent of E.


@lru_cache(maxsize=32)
def _kernel(index_type, info_eps, density_eps, backward, width, tangent=False):
    """One warp owns both independent LayerNorm reductions of one edge."""
    output = (
        "const float* dh, const float* dd,"
        if backward
        else "float* hidden, float* density,"
    )
    body = (
        """
        float scale = dd[row] * (1.f - d*d) * (2.f*raw);
        #pragma unroll
        for (int j=0; j<6; ++j) {
            float affine = v[j] * gain[32*j+lane] + shift[32*j+lane];
            float upstream = j<4 ? dh[WIDTH*row+32*j+lane] : scale*w[32*(j-4)+lane];
            u[j] = upstream * (sig[j] + affine*sig[j]*(1.f-sig[j])) * gain[32*j+lane];
        }
        float g1=sum(u[0]+u[1]+u[2]+u[3])*(1.f/128), g2=sum(u[4]+u[5])*(1.f/64);
        float t1=sum(u[0]*v[0]+u[1]*v[1]+u[2]*v[2]+u[3]*v[3])*(1.f/128);
        float t2=sum(u[4]*v[4]+u[5]*v[5])*(1.f/64);
        #pragma unroll
        for (int j=0; j<6; ++j)
            pre[192*row+32*j+lane] = j<4 ? r1*(u[j]-g1-v[j]*t1) : r2*(u[j]-g2-v[j]*t2);
        """
        if backward
        else """
        #pragma unroll
        for (int j=0; j<4; ++j) hidden[WIDTH*row+32*j+lane] = u[j];
        if (lane < WIDTH-128) hidden[WIDTH*row+128+lane] = float(lane == 0);
        if (!lane) density[row] = d;
        """
    )
    if tangent:
        output += "const float* pre_dot, float* hdot, float* ddot,"
        body += """
    float t[6];
    #pragma unroll
    for(int j=0;j<6;++j) t[j]=pre_dot[192*row+32*j+lane];
    float dm1=sum(t[0]+t[1]+t[2]+t[3])*(1.f/128), dm2=sum(t[4]+t[5])*(1.f/64);
    float dv1=sum(t[0]*v[0]+t[1]*v[1]+t[2]*v[2]+t[3]*v[3])*(1.f/128);
    float dv2=sum(t[4]*v[4]+t[5]*v[5])*(1.f/64);
    #pragma unroll
    for(int j=0;j<6;++j) {
        int c=32*j+lane; float act=v[j]*gain[c]+shift[c];
        t[j]=(t[j]-(j<4?dm1:dm2)-v[j]*(j<4?dv1:dv2))*(j<4?r1:r2)*gain[c]*
               sig[j]*(1.f+act*(1.f-sig[j]));
        if(j<4) hdot[128*row+c]=t[j];
    }
    float dd=sum(t[4]*w[lane]+t[5]*w[32+lane])*(2.f*raw)*(1.f-d*d);
    if(!lane) ddot[row]=dd;
    """
    return f"""
    using I = long long;
    using Index = {index_type};
    constexpr int WIDTH = {width};
    __device__ float sum(float x) {{
        #pragma unroll
        for (int i=16; i>=1; i>>=1) x+=__shfl_xor_sync(0xffffffff,x,i);
        return x;
    }}
    extern "C" __global__ void run(
        float* pre, const Index* sources, const Index* targets,
        const float* target_table, const float* source_table, const float* bias,
        const float* gain, const float* shift, const float* w, const float* b,
        {output} I count
    ) {{
        int lane=threadIdx.x%32;
        for (I row=(I(blockIdx.x)*blockDim.x+threadIdx.x)/32; row<count;
             row+=I(gridDim.x)*blockDim.x/32) {{
            float v[6],u[6],sig[6];
            I source=sources[row]*I(192), target=targets[row]*I(192);
            #pragma unroll
            for (int j=0; j<6; ++j) {{
                int col=32*j+lane;
                v[j]=((pre[192*row+col]+bias[col])+target_table[target+col])+source_table[source+col];
            }}
            float m1=sum(v[0]+v[1]+v[2]+v[3])*(1.f/128), m2=sum(v[4]+v[5])*(1.f/64);
            #pragma unroll
            for(int j=0;j<6;++j) v[j]-=j<4 ? m1 : m2;
            float r1=rsqrtf(sum(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3])*(1.f/128)+float({float(info_eps).hex()}));
            float r2=rsqrtf(sum(v[4]*v[4]+v[5]*v[5])*(1.f/64)+float({float(density_eps).hex()}));
            #pragma unroll
            for(int j=0;j<6;++j) {{
                v[j]*=j<4 ? r1 : r2;
                float a=v[j]*gain[32*j+lane]+shift[32*j+lane];
                sig[j]=1.f/(1.f+expf(-a));
                u[j]=a*sig[j];
            }}
            float raw=sum(u[4]*w[lane]+u[5]*w[32+lane])+b[0];
            float d=tanhf(raw*raw);
            {body}
        }}
    }}
    """


def _frozen_run(feat, source, target, packed, eps1, eps2, affine, grads=None):
    from equivariantx.ops.common.cuda.pointwise import launch, scalar_type

    edge, *rest = packed
    rows = feat.shape[0]
    width = 136 if affine else 128
    if grads is None:
        h, d = feat.new_empty((rows, width)), feat.new_empty((rows, 1))
    else:
        dx = torch.empty_like(feat, memory_format=torch.contiguous_format)
    code = _kernel(scalar_type(source.dtype), eps1, eps2, grads is not None, width)
    for begin in range(0, rows, _TILE):
        tile = slice(begin, min(begin + _TILE, rows))
        pre = feat[tile] @ edge
        args = [pre, source[tile], target[tile], *rest]
        if grads is None:
            args.extend((h[tile], d[tile]))
        else:
            args.extend(v[tile].contiguous() for v in grads)
        launch(code, args, (pre.shape[0],), 32 * pre.shape[0], threads=128)
        if grads is not None:
            torch.mm(pre, edge.T, out=dx[tile])
    return (h, d) if grads is None else dx


@torch.library.custom_op("eqx_oam_l::frozen_radial", mutates_args=())
def _frozen_heads(
    feat: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    packed: list[torch.Tensor],
    eps1: float,
    eps2: float,
    affine: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _frozen_run(feat, source, target, packed, eps1, eps2, affine)


@_frozen_heads.register_fake
def _fake_heads(feat, source, target, packed, eps1, eps2, affine=False):
    return feat.new_empty((feat.shape[0], 136 if affine else 128)), feat.new_empty(
        (feat.shape[0], 1)
    )


@torch.library.custom_op("eqx_oam_l::frozen_radial_vjp", mutates_args=())
def _frozen_vjp(
    feat: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    packed: list[torch.Tensor],
    eps1: float,
    eps2: float,
    dh: torch.Tensor,
    dd: torch.Tensor,
    affine: bool = False,
) -> torch.Tensor:
    return _frozen_run(feat, source, target, packed, eps1, eps2, affine, (dh, dd))


@_frozen_vjp.register_fake
def _fake_vjp(feat, source, target, packed, eps1, eps2, dh, dd, affine=False):
    return torch.empty_like(feat, memory_format=torch.contiguous_format)


def _context(ctx, inputs, output):
    feat, source, target, packed, ctx.eps1, ctx.eps2, ctx.affine = inputs
    ctx.save_for_backward(feat, source, target, *packed)


def _heads_backward(ctx, dh, dd):
    feat, source, target, *packed = ctx.saved_tensors
    if torch.is_grad_enabled():
        # Coordinate Hessians need a differentiable VJP. Restore the exact
        # live graph only when a caller actually requests a higher derivative.
        p = RadialParameters(*packed, None, None, ctx.eps1, ctx.eps2)
        h, d = _torch_heads(p, feat, source, target)
        (dx,) = torch.autograd.grad((h, d), feat, (dh[:, :128], dd), create_graph=True)
    else:
        dx = _frozen_vjp(
            feat, source, target, packed, ctx.eps1, ctx.eps2, dh, dd, ctx.affine
        )
    return dx, None, None, [None] * len(packed), None, None, None


_frozen_heads.register_autograd(_heads_backward, setup_context=_context)


def radial_heads(parameters, edge_feats, source_types, target_types, *, affine=False):
    """Return narrow hidden states and directed per-edge density contributions.

    `source_types` and `target_types` are already indexed by edges, allowing
    the caller to share the two integer arrays across interactions. They must
    be recomputed if species or connectivity changes.
    """
    packed = parameters[:8]
    if (
        edge_feats.is_cuda
        and torch.version.hip is None
        and edge_feats.dtype == torch.float32
        and not torch.is_autocast_enabled("cuda")
        and all(v.dtype == torch.float32 for v in packed)
        and source_types.dtype in (torch.int32, torch.int64)
        and source_types.dtype == target_types.dtype
        and not any(v.requires_grad for v in packed)
    ):
        return _frozen_heads(
            edge_feats,
            source_types.contiguous(),
            target_types.contiguous(),
            [v.contiguous() for v in packed],
            parameters.info_eps,
            parameters.density_eps,
            affine,
        )
    h, d = _torch_heads(parameters, edge_feats, source_types, target_types)
    if affine:
        tail = h.new_zeros((h.shape[0], 8))
        tail[:, 0] = 1.0
        h = torch.cat((h, tail), -1)
    return h, d


def affine_heads(parameters, edge_feats, source_types, target_types):
    """Return [E,136] hidden, aligned [136,P] projection, and [E,1] density.

    EQX's affine ABI requires a bias coordinate; its FP32 CUDA projection
    pads that coordinate to a multiple of eight. Emit the aligned form in
    the warp epilogue instead of materializing two full edge-wise copies.
    """
    h, d = radial_heads(parameters, edge_feats, source_types, target_types, affine=True)
    b = parameters.projection_bias.unsqueeze(0)
    projection = torch.cat((parameters.projection, b, b.new_zeros((7, b.shape[1]))))
    return h, projection, d


def layer_radial(inter, update, edge_feats, node_type, edge_index):
    """Return hidden, final projection, projection bias, tanh(density**2)."""
    p = prepare_radial(inter, update)
    source, target = edge_index
    h, d = radial_heads(p, edge_feats, node_type[source], node_type[target])
    return h, p.projection, p.projection_bias, d


def project(hidden, projection, bias):
    """Materialize final weights for a caller-selected edge tile."""
    return torch.addmm(bias, hidden, projection)
