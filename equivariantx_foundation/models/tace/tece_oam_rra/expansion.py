"""Local rotary-attention edge expansion and its input derivatives."""

from collections import defaultdict
from functools import lru_cache, partial

import torch

from equivariantx.ops.common.cuda.execution import execution_plan
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.program import Program

from .frame import (
    ENDPOINT_MATRIX,
    OUTPUT_FRAME,
    endpoint_adjoint,
    endpoint_matrix_adjoint,
    lower_output,
    output_frame,
)
from .layout import PairedFrameTile, paired_layout, paired_tile
from .score import score_roles


@lru_cache(maxsize=16)
def lower_expansion(base, saved):
    """Return the frozen expansion program for supported hidden and coefficient layouts."""
    nodes = base[0]
    _, coefficient, hidden = saved
    products = [
        i
        for i, (op, _, args, data) in enumerate(nodes)
        if op == "product"
        and len(args) == 6
        and args[:2] == (hidden, hidden)
        and data[1:] == (5, 0)
    ]
    if len(products) != 1:
        return None
    i = products[0]
    _, size, args, (metadata, _, _) = nodes[i]
    if metadata[:2] != (32, (110, 110, 20, 95, 5, 45)):
        return None
    program = Program(nodes)
    scales = []
    for arg, raw in zip(
        args[2:5],
        (program.slice(hidden, 0, 640), coefficient, program.slice(hidden, 640, 160)),
        strict=True,
    ):
        op, _, values, _ = nodes[arg]
        if op != "mul":
            return None
        constants = [k for k in values if nodes[k][0] == "constant"]
        if len(constants) != 1:
            return None
        scale = nodes[constants[0]][3]
        if program.scale(program.unary("silu", raw), scale) != arg:
            return None
        scales.append(scale)
    # Program.adjoint uses the widths to construct the two input adjoints.
    # Only ExpansionTile interprets this private, nonlinear product record.
    description = (
        32,
        (110, 95, 45),
        ("frozen_expansion", metadata, hidden, coefficient, tuple(scales)),
    )
    rows = list(nodes)
    rows[i] = ("product", size, (hidden, coefficient, args[5]), (description, 2, 0))
    return (lower_output(tuple(rows), base[2]), *base[1:])


@lru_cache(maxsize=16)
def outputs(metadata, scales, reverse):
    rows = metadata[2]
    terms = defaultdict(list)

    def factor(role, group):
        if role < 2:
            return f"h[{32 * group}]"
        if role == 2:
            return f"act(h[{32 * group}],{scales[0]:.17e}f)"
        if role == 3:
            return f"act(w[{32 * group}],{scales[1]:.17e}f)"
        if role == 4:
            return f"act(h[{32 * (20 + group)}],{scales[2]:.17e}f)"
        return f"dy[{32 * group}]"

    for indices, weight in rows:
        if (
            len(indices) != 6
            or not 0 <= indices[-1] < 45
            or any(k < -1 or k >= width for k, width in zip(indices, metadata[1]))
        ):
            raise ValueError("Invalid path index in the frozen edge expansion.")
        for role in range(5) if reverse else (5,):
            group = indices[role]
            if group < 0:
                continue
            if role == 3:
                group += 110
            elif role == 4:
                group += 20
            factors = [
                factor(i, k) for i, k in enumerate(indices) if i != role and k >= 0
            ]
            if weight != 1:
                factors.append(f"({float(weight):.17e}f)")
            terms[group].append((role, "*".join(factors) or "1.f"))
    cases = []
    for group in range(205 if reverse else 45):
        direct, chain = [], []
        for role, product in terms[group]:
            (chain if reverse and role in (2, 4) else direct).append(product)
        value = " + ".join(direct) or "0.f"
        if chain:
            scale = scales[0] if group < 20 else scales[2]
            value = (
                f"({value})+grad(h[{group * 32}],{scale:.17e}f)*({' + '.join(chain)})"
            )
        if reverse and group >= 110:
            k = (group - 110) * 32
            dest = f"dc[{k}]"
            value = f"grad(w[{k}],{scales[1]:.17e}f)*({value})"
        else:
            dest = f"{'dh' if reverse else 'z'}[{group * 32}]"
        cases.append((dest, value))
    return cases


@lru_cache(maxsize=16)
def source(metadata, scales, reverse):
    cases = [
        f"case {group}: {dest}={value};break;"
        for group, (dest, value) in enumerate(outputs(metadata, scales, reverse))
    ]
    parameters = "const float *dy, float *dh, float *dc" if reverse else "float *z"
    offsets = (
        "dy+=edge*1440+c;dh+=edge*3520+c;dc+=edge*3040+c;"
        if reverse
        else "z+=edge*1440+c;"
    )
    return f"""
    __device__ __forceinline__ float act(float x,float a){{return a*(x/(1.f+expf(-x)));}}
    __device__ __forceinline__ float grad(float x,float a){{float s=1.f/(1.f+expf(-x));return a*(s+x*s*(1.f-s));}}
    extern "C" __global__ void run(const float *h,const float *w,{parameters}){{
        long long edge=blockIdx.x; int c=threadIdx.x%32;
        h+=edge*3520+c; w+=edge*3040+c; {offsets}
        for(int i=threadIdx.x/32;i<{205 if reverse else 45};i+=blockDim.x/32){{
            switch(i){{{"".join(cases)}}}
        }}
    }}"""


@lru_cache(maxsize=16)
def dual_source(metadata, scales):
    terms = outputs(metadata, scales, True) + outputs(metadata, scales, False)
    cases = "".join(
        f"case {i}: {dest}=tangent({value}); break;"
        for i, (dest, value) in enumerate(terms)
    )
    return (
        """
    struct Dual {
        float x, d;
        __device__ Dual(float x=0.f, float d=0.f):x(x),d(d){}
    };
    __device__ __forceinline__ Dual operator+(Dual a,Dual b){return {a.x+b.x,a.d+b.d};}
    __device__ __forceinline__ Dual operator*(Dual a,Dual b){return {a.x*b.x,a.x*b.d+a.d*b.x};}
    __device__ __forceinline__ float tangent(Dual a){return a.d;}
    __device__ __forceinline__ Dual act(Dual a,float k){
        float s=1.f/(1.f+expf(-a.x)),g=s*(1.f-s);
        return {k*(a.x*s),k*(s+a.x*g)*a.d};
    }
    __device__ __forceinline__ Dual grad(Dual a,float k){
        float s=1.f/(1.f+expf(-a.x)),g=s*(1.f-s);
        return {k*(s+a.x*g),k*g*(2.f+a.x*(1.f-2.f*s))*a.d};
    }
    struct View {
        const float *x,*d;
        __device__ Dual operator[](int i) const {return {x[i],d ? d[i] : 0.f};}
    };
    extern "C" __global__ void run(const float* hx,const float* cx,const float* y,
        const float* vh,const float* vc,float* dh,float* dc,float* z){
        long long e=blockIdx.x; int c=threadIdx.x%32;
        View h={hx+e*3520+c,vh+e*3520+c}, w={cx+e*3040+c,vc+e*3040+c}, dy={y+e*1440+c,nullptr};
        dh+=e*3520+c; dc+=e*3040+c; z+=e*1440+c;
        for(int i=threadIdx.x/32;i<250;i+=blockDim.x/32){switch(i){
    """
        + cases
        + "}}}"
    )


def expansion_dual(metadata, scales, hidden, coefficient, grad, vh, vc):
    """Return the two input Hessian-vector products and the output tangent."""
    rows = hidden.shape[0]
    tensors = (hidden, coefficient, grad, vh, vc)
    if any(
        t.dtype != torch.float32 or t.device != hidden.device or t.shape != (rows, n)
        for t, n in zip(tensors, (3520, 3040, 1440, 3520, 3040), strict=True)
    ):
        raise ValueError(
            "Expected FP32 edge-expansion inputs with matching edge counts."
        )
    tensors = [t.contiguous() for t in tensors]
    result = tuple(torch.empty_like(t) for t in tensors[:3])
    if rows:
        code = dual_source(metadata, scales)
        kernel = kernels((code,), hidden.device)[code]
        runtime().launch(
            [(kernel, [t.data_ptr() for t in (*tensors, *result)], rows, 1, 256, 0)],
            torch.cuda.current_stream(hidden.device).cuda_stream,
        )
    return result


def reference_expansion(metadata, scales, hidden, coefficient):
    """Evaluate the independent local products with differentiable PyTorch operations."""
    h = hidden.reshape(-1, 110, 32)
    silu = torch.nn.functional.silu
    operands = (
        h,
        h,
        silu(h[:, :20]) * scales[0],
        silu(coefficient.reshape(-1, 95, 32)) * scales[1],
        silu(h[:, 20:25]) * scales[2],
    )
    rows = metadata[2]
    if not rows:
        return (
            hidden.sum(1, keepdim=True) * 0 + coefficient.sum(1, keepdim=True) * 0
        ).expand(-1, 1440)
    indices = torch.tensor([idx for idx, _ in rows], device=h.device)
    terms = h.new_tensor([w for _, w in rows]).reshape(1, -1, 1)
    for i, value in enumerate(operands):
        ids = indices[:, i]
        terms = terms * torch.where(
            ids[None, :, None] >= 0, value.index_select(1, ids.clamp_min(0)), 1
        )
    return (
        h.new_zeros((len(h), 45, 32))
        .index_add(1, indices[:, 5], terms)
        .reshape(-1, 1440)
    )


def expansion(metadata, scales, hidden, coefficient, grad=None):
    """Evaluate the FP32 expansion or its hidden- and coefficient-input adjoints."""
    rows = hidden.shape[0]
    if (
        metadata[:2] != (32, (110, 110, 20, 95, 5, 45))
        or len(scales) != 3
        or not hidden.is_cuda
        or hidden.dtype != torch.float32
        or hidden.shape != (rows, 3520)
        or coefficient.shape != (rows, 3040)
        or coefficient.dtype != hidden.dtype
        or coefficient.device != hidden.device
        or (
            grad is not None
            and (
                grad.shape != (rows, 1440)
                or grad.dtype != hidden.dtype
                or grad.device != hidden.device
            )
        )
    ):
        raise ValueError("Invalid input to the frozen FP32 edge expansion.")
    hidden, coefficient = hidden.contiguous(), coefficient.contiguous()
    reverse = grad is not None
    inputs = [hidden, coefficient]
    if reverse:
        inputs.append(grad.contiguous())
    outputs = (
        (torch.empty_like(hidden), torch.empty_like(coefficient))
        if reverse
        else (hidden.new_empty((hidden.size(0), 1440)),)
    )
    if hidden.size(0):
        code = source(metadata, scales, reverse)
        kernel = kernels((code,), hidden.device)[code]
        runtime().launch(
            [
                (
                    kernel,
                    [x.data_ptr() for x in (*inputs, *outputs)],
                    hidden.size(0),
                    1,
                    256,
                    0,
                )
            ],
            torch.cuda.current_stream(hidden.device).cuda_stream,
        )
    return outputs


class ExpansionTile(PairedFrameTile):
    """Evaluate one tile with a joint pullback through the frozen edge expansion."""

    def __init__(
        self,
        *args,
        roles,
        description,
        endpoint,
        matrix_endpoint,
        output_roles,
        pair_layout=None,
        score_layout=None,
    ):
        super().__init__(*args, pair_layout=pair_layout, score_layout=score_layout)
        self.roles = roles
        self.endpoint = endpoint
        self.matrix_endpoint = matrix_endpoint
        self.output_roles = output_roles
        _, self.expansion, self.hidden, self.coefficient, self.scales = description[2]

    def expansion_input(self, i):
        op, _, _, data = self.nodes[i]
        if op == "input" and data[1] == "edge":
            # A single requested input adjoint still reads both saved primals.
            return self.inputs[data[0]].flatten(1)[self.begin : self.end]
        return self.value(i)

    def value(self, i, destination=None):
        if i in self.output_roles:
            if i not in self.values:
                role = self.output_roles[i]
                x, w, g = self.nodes[i][2]
                a, b = (x, w) if role == 2 else (x, g) if role == 1 else (g, w)
                self.values[i] = output_frame(
                    self.value(a), self.value(b), role, destination
                )
                for child in self.dependencies["value", i]:
                    self.release(*child)
            return self.values[i]
        if i == self.endpoint or i == self.matrix_endpoint:
            if i not in self.values:
                x, w, *_ = self.nodes[i][2]
                if i == self.endpoint:
                    self.values[i] = endpoint_adjoint(
                        self.value(x), self.value(w), destination
                    )
                else:
                    self.values[i] = endpoint_matrix_adjoint(
                        self.inputs[0],
                        self.value(x),
                        self.source,
                        self.target,
                        destination,
                    )
                for child in self.dependencies["value", i]:
                    self.release(*child)
            return self.values[i]
        if i not in self.roles:
            return super().value(i, destination)
        if i not in self.values:
            hidden = self.expansion_input(self.hidden)
            coefficient = self.expansion_input(self.coefficient)
            role, other = self.roles[i]
            if role == 2:
                (self.values[i],) = expansion(
                    self.expansion, self.scales, hidden, coefficient
                )
                done = (i,)
            else:
                dy = self.value(self.nodes[i][2][2])
                grads = expansion(self.expansion, self.scales, hidden, coefficient, dy)
                self.values[i] = grads[role]
                done = [i]
                if other is not None and ("value", other) in self.dependencies:
                    self.values[other] = grads[1 - role]
                    done.append(other)
            for root in done:
                for child in self.dependencies["value", root]:
                    self.release(*child)
        return self.values[i]


def tile_factory(nodes):
    group = {
        i: (d[1], args[2], d[0])
        for i, (op, _, args, d) in enumerate(nodes)
        if op == "product" and d[0][2] and d[0][2][0] == "frozen_expansion"
    }
    if not group:
        return paired_tile(nodes)
    description = next(iter(group.values()))[2]
    if any(item[2] != description for item in group.values()):
        raise ValueError(
            "Expected a single frozen expansion per local attention layer."
        )
    gradients = {(role, seed): i for i, (role, seed, _) in group.items() if role != 2}
    roles = {
        i: (role, gradients.get((1 - role, seed)))
        for i, (role, seed, _) in group.items()
    }
    endpoint = next(
        (
            i
            for i, (op, _, _, d) in enumerate(nodes)
            if op == "product" and d[0][2] == ("frozen_endpoint_adjoint",)
        ),
        None,
    )
    return partial(
        ExpansionTile,
        roles=roles,
        description=description,
        endpoint=endpoint,
        matrix_endpoint=next(
            (
                i
                for i, (op, _, _, d) in enumerate(nodes)
                if op == "product" and d[0] == ENDPOINT_MATRIX
            ),
            None,
        ),
        output_roles={
            i: d[1]
            for i, (op, _, _, d) in enumerate(nodes)
            if op == "product" and d[0] == OUTPUT_FRAME
        },
        pair_layout=paired_layout(nodes),
        score_layout=score_roles(nodes),
    )


@lru_cache(maxsize=32)
def adjoint_tile(metadata):
    return tile_factory(execution_plan(metadata)[0])
