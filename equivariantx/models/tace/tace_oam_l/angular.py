"""OAM-L's 36 real component-normalized harmonics in one geometry pass.

Use the same exact Cartesian polynomial coefficients as the EQX harmonic
convolution. Evaluate geometry in FP64, cast to FP32, then multiply by the
FP32 cutoff as in the release. Radius is a separate *live* operand, since
other parts of the force/stress graph may already have computed it.
"""

from functools import lru_cache

import torch
from e3nn import o3

from equivariantx.ops.common.cuda.pointwise import launch, scalar_type
from equivariantx.utils.harmonics import polynomial_coefficients


def _reference(vectors, length, cutoff, dtype):
    u = vectors / length.clamp_min(1e-12)
    y = o3.spherical_harmonics(list(range(6)), u, False, "component").to(dtype)
    return y if cutoff is None else y * cutoff.to(dtype)


@lru_cache(maxsize=6)
def _source(cut_type, order):
    polys = [poly for l in range(6) for poly in polynomial_coefficients(l, "component")]
    lines = []
    # Common monomials, with the coordinate powers up to degree five shared
    # across all 36 components and derivative directions. No sampled tables.
    names = {(0, 0, 0): "1.0", (1, 0, 0): "x", (0, 1, 0): "y", (0, 0, 1): "z"}

    def monomial(p):
        if p not in names:
            axis = next(i for i in range(3) if p[i])
            parent = tuple(x - (axis == j) for j, x in enumerate(p))
            product = f"{'xyz'[axis]}*{monomial(parent)}"
            names[p] = "m" + "".join(map(str, p))
            lines.append(f"double {names[p]}={product};")
        return names[p]

    def expression(poly, *axes):
        terms = []
        for p, c in poly:
            for axis in axes:
                c *= p[axis]
                if not c:
                    break
                p = tuple(n - (axis == j) for j, n in enumerate(p))
            if c:
                terms.append(f"{float(c).hex()}*{monomial(p)}")
        return " + ".join(terms) or "0."

    lines.extend(("double ax=0., ay=0., az=0.;", "float ac=0.f;") if order else ())
    if order == 2:
        lines.extend(
            [
                "double tr=length[i]>=1.e-12 ? hr[i] : 0., h=float(hc[i]);",
                *(f"double t{a}=(hv[3*i+{k}]-{a}*tr)/r;" for k, a in enumerate("xyz")),
                "double bx=0., by=0., bz=0.;",
            ]
        )
    for col, poly in enumerate(polys):
        value = expression(poly)
        if order:
            lines.append(f"float g{col}=g[36*i+{col}];")
            if order == 1:
                lines.append(f"ac+=g{col}*float({value});")
            lines.append(f"double v{col}=double(g{col}*c);")
            tangent = []
            for axis, n in enumerate("xyz"):
                dv = expression(poly, axis)
                if dv != "0.":
                    lines.append(f"a{n}+=v{col}*({dv});")
                    if order == 2:
                        tangent.append(f"t{n}*({dv})")
                        terms = [
                            f"t{m}*({second})"
                            for j, m in enumerate("xyz")
                            if (second := expression(poly, axis, j)) != "0."
                        ]
                        hvp = " + ".join(terms) or "0."
                        lines.append(
                            f"b{n}+=double(g{col}*float(h))*({dv})+v{col}*({hvp});"
                        )
            if order == 2:
                dt = " + ".join(tangent) or "0."
                lines += [
                    f"float dot{col}=float({dt});",
                    f"ac+=g{col}*dot{col};",
                    f"dg[36*i+{col}]=float(h)*float({value})+c*dot{col};",
                ]
        else:
            lines.append(f"out[36*i+{col}]=float({value})*c;")
    if order == 1:
        lines += [
            "dv[3*i]=ax/r; dv[3*i+1]=ay/r; dv[3*i+2]=az/r;",
            "dr[i]=length[i]>=1.e-12 ? -(ax*x+ay*y+az*z)/r : 0.;",
            "dc[i]=C(ac);",
        ]
    elif order == 2:
        lines += [
            *(f"dv[3*i+{k}]=(b{a}-a{a}*tr/r)/r;" for k, a in enumerate("xyz")),
            "dr[i]=length[i]>=1.e-12 ? -((bx*x+by*y+bz*z)+(ax*tx+ay*ty+az*tz)-(ax*x+ay*y+az*z)*tr/r)/r : 0.;",
            "dc[i]=C(ac);",
        ]
    args = (
        "const float* g, const double* hv, const double* hr, const C* hc, double* dv, double* dr, C* dc, float* dg,"
        if order == 2
        else "const float* g, double* dv, double* dr, C* dc,"
        if order
        else "float* out,"
    )
    return f"""
    using I=long long; using C={cut_type};
    extern "C" __global__ void run(const double* vectors, const double* length,
        const C* cutoff, {args} I rows) {{
        for(I i=I(blockIdx.x)*blockDim.x+threadIdx.x; i<rows; i+=I(blockDim.x)*gridDim.x) {{
            double r=fmax(length[i], 1.e-12);
            double x=vectors[3*i]/r,y=vectors[3*i+1]/r,z=vectors[3*i+2]/r;
            float c=float(cutoff[i]);
            {chr(10).join(lines)}
        }}
    }}
    """


@torch.library.custom_op("eqx_oam_l_geometry::y5", mutates_args=())
def _y5(
    vectors: torch.Tensor, length: torch.Tensor, cutoff: torch.Tensor
) -> torch.Tensor:
    result = vectors.new_empty((vectors.shape[0], 36), dtype=torch.float32)
    launch(
        _source(scalar_type(cutoff.dtype), 0),
        (vectors, length, cutoff, result),
        (vectors.shape[0],),
        vectors.shape[0],
        threads=128,
    )
    return result


@_y5.register_fake
def _fake_y(vectors, length, cutoff):
    return vectors.new_empty((vectors.shape[0], 36), dtype=torch.float32)


@torch.library.custom_op("eqx_oam_l_geometry::y5_vjp", mutates_args=())
def _y5_vjp(
    vectors: torch.Tensor,
    length: torch.Tensor,
    cutoff: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = tuple(torch.empty_like(v) for v in (vectors, length, cutoff))
    launch(
        _source(scalar_type(cutoff.dtype), 1),
        (vectors, length, cutoff, gradient.contiguous(), *out),
        (vectors.shape[0],),
        vectors.shape[0],
        threads=128,
    )
    return out


@_y5_vjp.register_fake
def _fake_v(vectors, length, cutoff, gradient):
    return tuple(torch.empty_like(v) for v in (vectors, length, cutoff))


def _context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, gradient):
    return _y5_vjp(*ctx.saved_tensors, gradient)


@torch.library.custom_op("eqx_oam_l_geometry::y5_vjp_vjp", mutates_args=())
def _double_vjp(
    v: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    g: torch.Tensor,
    hv: torch.Tensor,
    hr: torch.Tensor,
    hc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = tuple(
        torch.empty_like(a, memory_format=torch.contiguous_format) for a in (v, r, c, g)
    )
    launch(
        _source(scalar_type(c.dtype), 2),
        (v, r, c, *(a.contiguous() for a in (g, hv, hr, hc)), *out),
        (len(v),),
        len(v),
        threads=128,
    )
    return out


@_double_vjp.register_fake
def _fake_double(v, r, c, g, hv, hr, hc):
    return tuple(
        torch.empty_like(a, memory_format=torch.contiguous_format) for a in (v, r, c, g)
    )


def _vjp_backward(ctx, hv, hr, hc):
    inputs = ctx.saved_tensors
    if torch.is_grad_enabled():

        def first(v, r, c, g):
            _, pullback = torch.func.vjp(
                lambda *args: _reference(*args, torch.float32), v, r, c
            )
            return pullback(g)

        _, pullback = torch.func.vjp(first, *inputs)
        return pullback((hv, hr, hc))
    return _double_vjp(*inputs, hv, hr, hc)


_y5.register_autograd(_backward, setup_context=_context)
_y5_vjp.register_autograd(_vjp_backward, setup_context=_context)


def edge_attributes(vectors, length, cutoff, model_dtype=torch.float32):
    """Return Y0..Y5 times cutoff with the release's input and cast conventions."""
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 3
        or length.shape != (vectors.shape[0], 1)
    ):
        raise ValueError("Expected vectors [E,3], length [E,1]")
    if cutoff is not None and cutoff.shape != length.shape:
        raise ValueError("Expected cutoff [E,1]")
    if (
        vectors.is_cuda
        and torch.version.hip is None
        and vectors.dtype == torch.float64
        and length.dtype == torch.float64
        and model_dtype == torch.float32
        and cutoff is not None
        and cutoff.dtype in (torch.float32, torch.float64)
    ):
        return _y5(vectors.contiguous(), length.contiguous(), cutoff.contiguous())
    return _reference(vectors, length, cutoff, model_dtype)
