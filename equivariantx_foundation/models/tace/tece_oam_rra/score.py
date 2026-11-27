"""Frozen four-head rotary score on projected angular blocks."""

import math
from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import compact_metadata, parse_metadata
from equivariantx.utils.program import Program

from .expressions import rotary_dot

WIDTHS = (320, 512, 384, 256, 128)
TAG = "frozen_rotary_score"


@lru_cache(maxsize=2)
def template(reverse):
    p = Program()
    widths = (*WIDTHS, *WIDTHS, 320, 320, 8, 8, 4, 4, 4)
    xs = [
        p.input(i, "edge" if i < 10 or i in (12, 16) else "shared", width)
        for i, width in enumerate(widths)
    ]
    q, k = [
        p.concatenate((p.binary("add", xs[j], xs[b]), *xs[j + 1 : j + 5]))
        for j, b in ((0, 10), (5, 11))
    ]
    radial = p.binary("add", xs[12], xs[13])
    phase = p.scale(p.unary("tanh", p.slice(radial, 4, 4)), math.pi)
    raw = rotary_dot(p, q, k, phase, 5, 4, 64, 4)
    score = p.binary(
        "add",
        p.binary("mul", p.binary("mul", raw, xs[14]), xs[15]),
        p.slice(radial, 0, 4),
    )
    if not reverse:
        return p.nodes, (score,)
    grad = {slot: i for i, slot, _ in p.adjoint((score,), (xs[16],), (*range(10), 12))}
    return p.nodes, tuple(grad[slot] for slot in (*range(10), 12))


@lru_cache(maxsize=32)
def score_nodes(nodes, outputs, reverse):
    """Replace an exact rotary score or its complete frozen input pullback."""
    pattern, roots = template(reverse)
    anchor = roots[-1]
    lookup = {node: i for i, node in enumerate(nodes)}

    def match(i, j, bound):
        if i in bound:
            return bound[i] == j
        op, width, args, data = pattern[i]
        actual, size, children, desc = nodes[j]
        if width != size or (
            op != "input"
            and (op != actual or desc != data or len(args) != len(children))
        ):
            return False
        bound[i] = j
        return op == "input" or all(match(a, b, bound) for a, b in zip(args, children))

    def resolve(i, bound):
        if i not in bound:
            op, width, args, data = pattern[i]
            if op == "input":
                return -1
            children = tuple(resolve(a, bound) for a in args)
            bound[i] = lookup.get((op, width, children, data), -1)
        return bound[i]

    independent = []
    for op, _, args, d in nodes:
        independent.append(
            d[1] == "shared" if op == "input" else all(independent[j] for j in args)
        )
    candidates = range(len(nodes)) if reverse else outputs
    for j in candidates:
        if nodes[j][:2] != pattern[anchor][:2]:
            continue
        bound = {}
        if not match(anchor, j, bound):
            continue
        actual = tuple(resolve(i, bound) for i in roots)
        if -1 in actual or any(i not in bound for i in range(17 if reverse else 16)):
            continue
        if nodes[bound[15]][0] != "constant" or not all(
            independent[bound[i]] for i in (10, 11, 13, 14)
        ):
            continue
        inputs = tuple(bound[i] for i in (*range(15), *((16,) if reverse else ())))
        if (
            any(independent[i] for i in inputs[:10])
            or independent[bound[12]]
            or (reverse and independent[bound[16]])
        ):
            continue
        # A partial pullback should only compute the requested derivatives.
        live, pending = set(), list(outputs)
        while pending:
            i = pending.pop()
            if i not in live:
                live.add(i)
                pending.extend(nodes[i][2])
        if not set(actual).issubset(live):
            continue
        new = list(nodes)
        for role, i in enumerate(actual):
            size = nodes[i][1]
            dims = tuple(nodes[k][1] for k in inputs) + (size,)
            desc = (1, dims, (TAG, reverse, role, nodes[bound[15]][3]))
            new[i] = ("product", size, (*inputs, inputs[0]), (desc, len(inputs), 0))
        return tuple(new)
    return nodes


@lru_cache(maxsize=32)
def lower_primal(base):
    nodes = score_nodes(base[0], (base[1],), False)
    return base if nodes is base[0] else (nodes, *base[1:])


@lru_cache(maxsize=32)
def lower_pullback(metadata):
    nodes, outputs = parse_metadata(metadata)
    lowered = score_nodes(nodes, tuple(i for i, _, _ in outputs), True)
    return metadata if nodes is lowered else compact_metadata(repr((lowered, outputs)))


@lru_cache(maxsize=32)
def score_roles(nodes):
    return {
        i: d[0][2][1:]
        for i, (op, _, _, d) in enumerate(nodes)
        if op == "product" and d[0][2] and d[0][2][0] == TAG
    }


@lru_cache(maxsize=16)
def score_source(reverse, scale):
    args = [f"const float* {name}{m}" for name in ("q", "k") for m in range(5)]
    args += [
        "const float* qb",
        "const float* kb",
        "const float* rad",
        "const float* rb",
        "const float* tau",
    ]
    if reverse:
        args += ["const float* seed"] + [
            f"float* d{name}{m}" for name in ("q", "k") for m in range(5)
        ]
    args.append("float* out")
    body = []
    for m, width in enumerate(WIDTHS):
        n = 5 - m
        section = f"int a=e*{width}+j*64+c;"
        if m:
            section += f"""
            int b=a+{n * 64};
            float qr=q{m}[a],qi=q{m}[b],kr=k{m}[a],ki=k{m}[b],co=C[{m - 1}][h],si=S[{m - 1}][h];
            """
            section += (
                f"""
                dq{m}[a]=g[h]*(co*kr-si*ki); dq{m}[b]=g[h]*(co*ki+si*kr);
                dk{m}[a]=g[h]*(co*qr+si*qi); dk{m}[b]=g[h]*(co*qi-si*qr);
                acc+={m}*(qr*(-si*kr-co*ki)+qi*(co*kr-si*ki));
            """
                if reverse
                else "acc+=qr*(co*kr-si*ki)+qi*(co*ki+si*kr);"
            )
        else:
            section += "float qr=q0[a]+qb[j*64+c],kr=k0[a]+kb[j*64+c];"
            section += "dq0[a]=g[h]*kr;dk0[a]=g[h]*qr;" if reverse else "acc+=qr*kr;"
        body.append(f"for(int j=t/64;j<{n};j+=2){{{section}}}")
    result = (
        "out[e*8+t]=seed[e*4+t];out[e*8+4+t]=((sum*g[t])*pi)*(1.f-z[t]*z[t]);"
        if reverse
        else "out[e*4+t]=(sum*tau[t])*scale+(rad[e*8+t]+rb[t]);"
    )
    gain = "g[h]=(seed[e*4+h]*scale)*tau[h];" if reverse else ""
    return f"""
    extern "C" __global__ void run({",".join(args)}) {{
      constexpr float pi=3.14159265358979323846f,scale={float(scale).hex()}f;
      int t=threadIdx.x,c=t%64,h=c/16, e=blockIdx.x;
      __shared__ float C[4][4],S[4][4],partial[2][4],z[4],g[4];
      if(t<16) {{
        int m=t/4,h=t%4; float a=tanhf(rad[e*8+4+h]+rb[4+h]);
        sincosf((m+1)*(pi*a),S[m]+h,C[m]+h);
        if(!m){{z[h]=a;{gain}}}
      }}
      __syncthreads();
      float acc=0.f;
      {"".join(body)}
      for(int s=8;s;s/=2) acc+=__shfl_down_sync(0xffffffff,acc,s,16);
      if(c%16==0) partial[t/64][h]=acc;
      __syncthreads();
      if(t<4){{float sum=partial[0][t]+partial[1][t];{result}}}
    }}"""


def score_outputs(inputs, scale, reverse):
    """Evaluate the score or its projected-block and radial derivatives."""
    q = inputs[0]
    outs = (
        [q.new_empty((len(q), width)) for width in (*WIDTHS, *WIDTHS)]
        if reverse
        else []
    )
    outs.append(q.new_empty((len(q), 8 if reverse else 4)))
    if len(q):
        code = score_source(reverse, scale)
        ptrs = [v.data_ptr() for v in (*inputs, *outs)]
        runtime().launch(
            [(kernels((code,), q.device)[code], ptrs, len(q), 1, 128, 0)],
            torch.cuda.current_stream(q.device).cuda_stream,
        )
    return outs


def tile_score(tile, i, destination=None):
    if i not in tile.values:
        reverse, _, scale = tile.score_roles[i]
        values = [tile.value(j).contiguous() for j in tile.nodes[i][2][:-1]]
        out = score_outputs(values, scale, reverse)
        for j, (_, role, _) in tile.score_roles.items():
            if ("value", j) in tile.dependencies:
                tile.values[j] = out[role]
                for child in tile.dependencies["value", j]:
                    tile.release(*child)
    return tile.values[i] if destination is None else destination.copy_(tile.values[i])
