"""Frozen degree-wise channel projections for the 7M bilinear product."""

from ast import literal_eval
from functools import lru_cache

import torch

from equivariantx.mbe.ace.bilinear import BilinearContraction
from equivariantx.ops.common.cuda.pointwise import launch
from tace.models.spherical.tece_oam_rra import BilinearMoEACE


def angular_plan(tp):
    tensors = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
    if sorted(ins.i_out for ins in tp.instructions) != list(range(len(tp.irreps_out))):
        raise ValueError("Expected one gated path per output block.")
    channels = tp.irreps_out[0].mul
    rows = []
    for i, ins in enumerate(tp.instructions):
        blocks = tuple(
            (reps.slices()[j].start, reps[j].ir.dim)
            for reps, j in zip(tensors, (ins.i_in1, ins.i_in2, ins.i_out))
        )
        if not (
            ins.connection_mode == "uuu"
            and ins.has_weight
            and ins.path_shape == (channels,)
        ):
            raise ValueError("Expected channel-wise bilinear paths.")
        rows.append((blocks, BilinearContraction.coupling_entries(tp, i)))
    return repr((tuple(r.dim for r in tensors), rows))


@lru_cache(maxsize=16)
def unpack(meta):
    dimensions, rows = literal_eval(meta)
    angular_width = sum(blocks[2][1] for blocks, _ in rows)
    return dimensions, rows, dimensions[2] // angular_width


@lru_cache(maxsize=16)
def source(meta, backward):
    (a, b, c), rows, channels = unpack(meta)
    args = "const float* X,const float* B,const float* T,"
    args += "const float* G,float* DX,float* DB,float* DT," if backward else "float* Z,"
    body = [
        f"for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n*{channels};i+=(long long)gridDim.x*blockDim.x){{",
        f"int lane=i%{channels}; long long row=i/{channels};",
        f"const float *x=X+row*{a}, *b=B+row*{b}, *t=T+row*{channels * len(rows)}+lane;",
    ]
    body.append(
        f"const float* g=G+row*{c};float *dx=DX+row*{a},*db=DB+row*{b},*dt=DT+row*{channels * len(rows)}+lane;"
        if backward
        else f"float* z=Z+row*{c};"
    )
    used = set()
    for index, (blocks, cg) in enumerate(rows):
        (so, di), (bo, dj), (zo, dk) = blocks
        for v, start, d, columns in (
            ("x", so, di, set(r[0] for r in cg)),
            ("b", bo, dj, set(r[1] for r in cg)),
        ):
            for k in sorted(columns):
                key = (v, start, d, k)
                if key not in used:
                    name = f"{v}{start}_{k}"
                    body.append(f"float {name}={v}[{start}+lane*{d}+{k}];")
                    if backward:
                        body.append(f"float d{name}=0.f;")
                    used.add(key)
        body.extend(("{", f"float gate=t[{channels * index}];"))
        if backward:
            body.append("float gt=0.f;")
        for k in range(dk):
            body.append(
                "{"
                + (f"float adj=g[{zo}+lane*{dk}+{k}], gg=gate*adj;" if backward else "")
            )
            body.append("float u=0.f;")
            for i, j, _, co in (r for r in cg if r[2] == k):
                v, w = f"x{so}_{i}", f"b{bo}_{j}"
                coef = f"float({float(co).hex()})"
                body.append(f"u=fmaf({v},{coef}*{w},u);")
                if backward:
                    body.extend(
                        (
                            f"d{v}=fmaf(gg*{coef},{w},d{v});",
                            f"d{w}=fmaf(gg*{coef},{v},d{w});",
                        )
                    )
            body.extend(
                (
                    "gt=fmaf(adj,u,gt);"
                    if backward
                    else f"z[{zo}+lane*{dk}+{k}]=gate*u;",
                    "}",
                )
            )
        if backward:
            body.append(f"dt[{channels * index}]=gt;")
        body.append("}")
    if backward:
        for v, start, d, k in sorted(used):
            body.append(f"d{v}[{start}+lane*{d}+{k}]=d{v}{start}_{k};")
    return (
        'extern "C" __global__ void run('
        + args
        + "long long n){"
        + "".join(body)
        + "}}"
    )


@torch.library.custom_op("eqx_tace_7m::angular", mutates_args=(), device_types="cuda")
def angular(
    x: torch.Tensor, b: torch.Tensor, t: torch.Tensor, meta: str, size: int
) -> torch.Tensor:
    z = x.new_empty((x.size(0), size))
    launch(
        source(meta, False),
        (x.contiguous(), b.contiguous(), t.contiguous(), z),
        (x.size(0),),
        x.size(0) * unpack(meta)[2],
    )
    return z


@angular.register_fake
def fake(x, b, t, meta, size):
    return x.new_empty((x.size(0), size))


@torch.library.custom_op(
    "eqx_tace_7m::angular_vjp", mutates_args=(), device_types="cuda"
)
def vjp(
    x: torch.Tensor, b: torch.Tensor, t: torch.Tensor, g: torch.Tensor, meta: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Output scalars can use only a subset of input blocks.
    outs = [
        torch.zeros_like(v, memory_format=torch.contiguous_format) for v in (x, b, t)
    ]
    launch(
        source(meta, True),
        (x.contiguous(), b.contiguous(), t.contiguous(), g.contiguous(), *outs),
        (x.size(0),),
        x.size(0) * unpack(meta)[2],
    )
    return tuple(outs)


@vjp.register_fake
def fake_vjp(x, b, t, g, meta):
    return tuple(torch.empty_like(a) for a in (x, b, t))


def setup(ctx, inputs, output):
    x, b, t, ctx.meta, _ = inputs
    ctx.save_for_backward(x, b, t)


def reverse(ctx, g):
    return (*vjp(*ctx.saved_tensors, g, ctx.meta), None, None)


angular.register_autograd(reverse, setup_context=setup)


def matrix_blocks(linear):
    instructions = linear.linear.instructions
    weight = linear.weight
    if not isinstance(weight, torch.nn.ParameterList):
        weight = weight.split([p.path_shape[0] * p.path_shape[1] for p in instructions])
    return {
        p.i_out: (p.i_in, w.detach().double().reshape(p.path_shape) * p.path_weight)
        for p, w in zip(instructions, weight, strict=True)
    }


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
