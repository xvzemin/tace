"""Projected channelwise functions on two- and four-sheet Fourier grids."""

from functools import lru_cache
from math import isfinite

import torch

from ...common.cuda.pointwise import launch, scalar_type
from ...precision import validate_uniform


@lru_cache(maxsize=128)
def transform_source(dtype, inputs, outputs, tile, transpose_x, transpose_m):
    coordinate = "int k=z%K,j=z/K;" if transpose_x else "int k=z/TILE,j=z%TILE;"
    offset = "(row*C+c)*K+k" if transpose_x else "(row*K+k)*C+c"
    matrix = "m[k*N+j]" if transpose_m else "m[j*K+k]"
    return f"""
    using T={scalar_type(dtype)}; using I=long long;
    extern "C" __global__ void run(const T* x, const T* m,T* y,I rows,I C) {{
        constexpr int K={inputs},N={outputs},TILE={tile};
        __shared__ T v[K*TILE];
        int tid=threadIdx.x,lane=tid%TILE;
        I tiles=(C+TILE-1)/TILE;
        for(I block=blockIdx.x;block<rows*tiles;block+=gridDim.x) {{
            I row=block/tiles,lo=block%tiles*TILE;
            for(int z=tid; z<K*TILE; z+=blockDim.x) {{
                {coordinate} I c=lo+j;
                v[k*TILE+j]=c<C?x[{offset}]:T(0);
            }}
            __syncthreads();
            I c=lo+lane;
            for(int j=tid/TILE;j<N;j+=blockDim.x/TILE) {{
                T sum=0;
                #pragma unroll
                for(int k=0;k<K;++k) sum += {matrix}*v[k*TILE+lane];
                if(c<C)y[(row*N+j)*C+c]=sum;
            }}
            __syncthreads();
        }}
    }}"""


@torch.library.custom_op("eqx::o2_grid_transform", mutates_args=(), device_types="cuda")
def transform(matrix: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply a fixed Fourier band matrix to every channel."""
    validate_uniform((matrix, x))
    if matrix.ndim != 2 or x.ndim < 2 or matrix.shape[1] != x.shape[-2]:
        raise ValueError("Expected a band matrix and matching coefficient bands.")
    n, k = matrix.shape
    c = x.shape[-1]
    out = x.new_empty((*x.shape[:-2], n, c))
    if not out.numel():
        return out
    if not k:
        return out.zero_()
    if not matrix.is_contiguous() and not matrix.T.is_contiguous():
        matrix = matrix.contiguous()
    if not x.is_contiguous() and not x.transpose(-1, -2).is_contiguous():
        x = x.contiguous()
    tile = min(32, 1 << (c - 1).bit_length())
    while tile > 1 and k * tile * x.element_size() > 32768:
        tile //= 2
    if k * tile * x.element_size() > 49152:
        raise ValueError("Fourier band matrix exceeds shared memory.")
    source = transform_source(
        x.dtype, k, n, tile, not x.is_contiguous(), not matrix.is_contiguous()
    )
    rows = out.numel() // (n * c)
    threads = min(256, max(32, tile * 8))
    launch(
        source,
        (x, matrix, out),
        (rows, c),
        rows * ((c + tile - 1) // tile) * threads,
        threads=threads,
    )
    return out


@transform.register_fake
def fake_transform(matrix, x):
    return x.new_empty((*x.shape[:-2], matrix.shape[0], x.shape[-1]))


def transform_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def transform_backward(ctx, gradient):
    return None, transform(ctx.saved_tensors[0].T, gradient)


transform.register_autograd(transform_backward, setup_context=transform_context)


def sigmoid_derivative(order, tanh=False):
    """Return the derivative as a polynomial in sigmoid or tanh."""
    if order == 0:
        return "s"
    # s' = q(s). Higher derivatives are q(s) p(s); p <- (q p)'.
    q = (1, 0, -1) if tanh else (0, 1, -1)
    p = [1]
    for _ in range(1, order):
        a = [0] * (len(p) + 2)
        for i, x in enumerate(p):
            for j, y in enumerate(q):
                a[i + j] += x * y
        p = [i * a[i] for i in range(1, len(a))]
    expression = f"T({float(p[-1])!r})"
    for c in p[-2::-1]:
        expression = f"(({expression})*s+T({float(c)!r}))"
    return f"((T(1)-s*s)*({expression}))" if tanh else f"(s*(T(1)-s)*({expression}))"


def activation_source(kind, order, degree, bound):
    if kind == "product":
        return "T result=v0;"
    if kind == "square":
        value = ("v0*v0", "T(2)*v0", "T(2)")[order] if order < 3 else "T(0)"
        return f"T result={value};"
    if kind == "relu":
        value = ("v0<=0?T(0):v0", "v0<=0?T(0):T(1)")[order] if order < 2 else "T(0)"
        return f"T result={value};"
    if kind == "softplus":
        if order == 0:
            return "T result=v0>T(20)?v0:log1p(exp(v0));"
        if order == 1:
            return "T result=v0>T(20)?T(1):T(1)/(T(1)+exp(-v0));"
        return f"T s=T(1)/(T(1)+exp(-v0)); T result=v0<T(20)?{sigmoid_derivative(order - 1)}:T(0);"
    if kind == "gelu":
        cdf = "T cdf=T(0.5)*erfc(-v0*T(0.70710678118654752440));"
        pdf = "T pdf=T(0.39894228040143267794)*exp(-T(0.5)*v0*v0);"
        if order == 0:
            return cdf + "T result=v0*cdf;"
        if order == 1:
            return cdf + pdf + "T result=cdf+v0*pdf;"
        # d^n(x Phi) = x Phi^(n) + n Phi^(n-1); derivatives of the
        # normal density are Hermite polynomials times the same density.
        return (
            pdf
            + f"""
          T prev=T(1), cur=v0;
          #pragma unroll
          for(int j=1;j<{order - 1};++j) {{T next=v0*cur-T(j)*prev; prev=cur; cur=next;}}
          T result=T({(-1) ** order})*(T({order})*prev-v0*cur)*pdf;"""
        )
    if kind == "sin":
        return (
            "T result="
            + ("sin(v0)", "cos(v0)", "-sin(v0)", "-cos(v0)")[order % 4]
            + ";"
        )
    if kind in ("tanh", "sigmoid", "silu"):
        init = "T s=tanh(v0);" if kind == "tanh" else "T s=T(1)/(T(1)+exp(-v0));"
        value = sigmoid_derivative(order, tanh=kind == "tanh")
        if kind == "silu":
            value = f"v0*{value}" + (
                f"+T({order})*{sigmoid_derivative(order - 1)}" if order else ""
            )
        return f"{init} T result={value};"
    if kind != "polynomial":
        raise ValueError("Unsupported projected activation.")
    if order > degree:
        return "T result=0;"
    # Differentiate the same Clenshaw recurrence, not a monomial approximation.
    return f"""
        T a[{order + 1}]={{}}, b[{order + 1}]={{}}, t=v0/T({bound!r});
        #pragma unroll
        for(int j={degree}; j>0; --j) {{
            #pragma unroll
            for(int k={order}; k>=0; --k) {{
                T u=T(2)*t*a[k]-b[k]+(k ? T(2*k)*a[k-1] : coeff[j]);
                b[k]=a[k]; a[k]=u;
            }}
        }}
        T result=t*a[{order}]-b[{order}]+{"coeff[0]" if not order else f"T({order})*a[{order - 1}]"};
        #pragma unroll
        for(int k=0; k<{order}; ++k) result/=T({bound!r});
    """


@lru_cache(maxsize=256)
def kernel_source(dtype, f, g, tile, modes, target, kind, order, degree, bound, paired):
    count = len(modes)
    args = ", ".join(f"const T* x{i}" for i in range(count))
    load = "".join(
        f"bands[{i}*F*TILE+j*TILE+lane]=c<C ? x{i}[(row*F+j)*C+c] : T(0);"
        for i in range(count)
    )
    dot = "".join(
        f"v{i} += {'b' if mode else 'a'}[k*F+j]*bands[{i}*F*TILE+j*TILE+lane];"
        for i, mode in enumerate(modes)
    )
    acc = "".join(f"T v{i}=0;" for i in range(count))
    if paired:
        point = "grid[k*TILE+lane]=v0*v2; grid[(G+k)*TILE+lane]=v0*v1;"
        targets = modes[1:]
    else:
        point = activation_source(kind, order, degree, bound)
        point += "".join(f"result *= v{i};" for i in range(1, count))
        point += "grid[k*TILE+lane]=result;"
        targets = (target,)
    sums = "".join(f"T sum{i}=0;" for i in range(len(targets)))
    project = "".join(
        f"sum{i} += {'b' if dst else 'a'}[k*F+j]*grid[({i}*G+k)*TILE+lane];"
        for i, dst in enumerate(targets)
    )
    store = "".join(f"out{i}[(row*F+j)*C+c]=sum{i};" for i in range(len(targets)))
    dest = ", ".join(f"T* out{i}" for i in range(len(targets)))
    return f"""
    using T={scalar_type(dtype)}; using I=long long;
    extern "C" __global__ void run(const T* a,const T* b,{args},
        {"const T* coeff," if degree >= 0 else ""} {dest}, I rows, I C) {{
        constexpr int F={f},G={g},TILE={tile};
        __shared__ T bands[{count}*F*TILE], grid[{len(targets)}*G*TILE];
        int lane=threadIdx.x%TILE, w=threadIdx.x/TILE, nw=blockDim.x/TILE;
        I tiles=(C+TILE-1)/TILE;
        for(I block=blockIdx.x;block<rows*tiles;block+=gridDim.x) {{
            I row=block/tiles,c=(block%tiles)*TILE+lane;
            for(int j=w;j<F;j+=nw){{ {load} }}
            __syncthreads();
            for(int k=w;k<G;k+=nw){{
                {acc}
                #pragma unroll
                for(int j=0;j<F;++j){{ {dot} }}
                {point}
            }}
            __syncthreads();
            for(int j=w;j<F;j+=nw){{
                {sums}
                #pragma unroll
                for(int k=0;k<G;++k) {{ {project} }}
                if(c<C) {{ {store} }}
            }}
            __syncthreads();
        }}
    }}"""


def launch_projection(
    synthesis,
    analysis,
    coefficients,
    values,
    modes,
    target,
    kind,
    order,
    bound,
    paired=False,
):
    validate_uniform((values, synthesis, analysis, coefficients))
    if (
        not values
        or synthesis.ndim != 2
        or analysis.shape != synthesis.shape
        or values[0].ndim < 2
        or any(v.shape != values[0].shape for v in values)
        or values[0].shape[-2] != synthesis.shape[1]
        or len(modes) != len(values)
        or any(i not in (0, 1) for i in (*modes, target))
        or order < 0
        or (coefficients is not None) != (kind == "polynomial")
        or (
            coefficients is not None
            and (coefficients.ndim != 1 or not coefficients.numel())
        )
        or not isfinite(bound)
        or bound <= 0
    ):
        raise ValueError("Invalid coefficient layout or grid function.")
    g, f = synthesis.shape
    c = values[0].shape[-1]
    output = values[0].new_empty(values[0].shape)
    outputs = (output, torch.empty_like(output)) if paired else (output,)
    if not output.numel():
        return outputs if paired else output
    tile = min(32, 1 << (c - 1).bit_length())
    elements = len(values) * f + (2 if paired else 1) * g
    while tile > 1 and tile * elements * output.element_size() > 32768:
        tile //= 2
    if tile * elements * output.element_size() > 49152:
        raise ValueError("Grid projection exceeds shared memory.")
    degree = -1 if coefficients is None else coefficients.numel() - 1
    source = kernel_source(
        output.dtype,
        f,
        g,
        tile,
        tuple(modes),
        target,
        kind,
        order,
        degree,
        bound,
        paired,
    )
    operands = (synthesis, analysis, *values) + (
        () if coefficients is None else (coefficients,)
    )
    rows = values[0].numel() // (f * c)
    threads = min(256, max(32, tile * 8))
    launch(
        source,
        (*[v.contiguous() for v in operands], *outputs),
        (rows, c),
        rows * ((c + tile - 1) // tile) * threads,
        threads=threads,
    )
    return outputs if paired else output


@torch.library.custom_op("eqx::o2_grid_project", mutates_args=(), device_types="cuda")
def project(
    synthesis: torch.Tensor,
    analysis: torch.Tensor,
    coefficients: torch.Tensor | None,
    values: list[torch.Tensor],
    modes: list[int],
    target: int,
    kind: str,
    order: int,
    bound: float,
) -> torch.Tensor:
    """Apply a grid function times linear factors, followed by a fixed projection."""
    return launch_projection(
        synthesis, analysis, coefficients, values, modes, target, kind, order, bound
    )


@project.register_fake
def fake_project(
    synthesis, analysis, coefficients, values, modes, target, kind, order, bound
):
    return torch.empty_like(values[0], memory_format=torch.contiguous_format)


def setup_context(ctx, inputs, output):
    synthesis, analysis, coeff, values, modes, target, kind, order, bound = inputs
    ctx.save_for_backward(synthesis, analysis, coeff, *values)
    ctx.modes, ctx.target, ctx.kind, ctx.order, ctx.bound = (
        tuple(modes),
        target,
        kind,
        order,
        bound,
    )


def backward(ctx, grad):
    synthesis, analysis, coeff, *values = ctx.saved_tensors
    if ctx.kind == "product" and len(values) == 2 and all(ctx.needs_input_grad[3]):
        grads = product_adjoint(
            synthesis, analysis, grad, *values, *ctx.modes, ctx.target
        )
        return None, None, None, list(grads), None, None, None, None, None
    grads = []
    for i, needed in enumerate(ctx.needs_input_grad[3]):
        if not needed:
            grads.append(None)
            continue
        modes, args, order = list(ctx.modes), list(values), ctx.order
        if i == 0 and ctx.kind != "product":
            modes.append(ctx.target)
            args.append(grad)
            order += 1
        else:
            modes[i] = ctx.target
            args[i] = grad
        grads.append(
            project(
                synthesis,
                analysis,
                coeff,
                args,
                modes,
                ctx.modes[i],
                ctx.kind,
                order,
                ctx.bound,
            )
        )
    return None, None, None, grads, None, None, None, None, None


project.register_autograd(backward, setup_context=setup_context)


@torch.library.custom_op(
    "eqx::o2_grid_product_adjoint", mutates_args=(), device_types="cuda"
)
def product_adjoint(
    synthesis: torch.Tensor,
    analysis: torch.Tensor,
    g: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    mx: int,
    my: int,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project both product cotangents using a single synthesis of their common factor."""
    return launch_projection(
        synthesis,
        analysis,
        None,
        [g, x, y],
        [target, mx, my],
        mx,
        "product",
        0,
        1.0,
        True,
    )


@product_adjoint.register_fake
def fake_adjoint(synthesis, analysis, g, x, y, mx, my, target):
    return x.new_empty(x.shape), y.new_empty(y.shape)


def adjoint_context(ctx, inputs, output):
    a, b, g, x, y, ctx.mx, ctx.my, ctx.target = inputs
    ctx.save_for_backward(a, b, g, x, y)


def adjoint_backward(ctx, p, q):
    a, b, g, x, y = ctx.saved_tensors
    mx, my, t = ctx.mx, ctx.my, ctx.target

    def pair(u, v, i, j, out):
        return project(a, b, None, [u, v], [i, j], out, "product", 0, 1.0)

    dg = (
        pair(p, y, mx, my, t) + pair(x, q, mx, my, t)
        if ctx.needs_input_grad[2]
        else None
    )
    if all(ctx.needs_input_grad[3:5]):
        dx, dy = product_adjoint(a, b, g, p, q, mx, my, t)
    else:
        dx = pair(g, q, t, my, mx) if ctx.needs_input_grad[3] else None
        dy = pair(g, p, t, mx, my) if ctx.needs_input_grad[4] else None
    return None, None, dg, dx, dy, None, None, None


product_adjoint.register_autograd(adjoint_backward, setup_context=adjoint_context)
