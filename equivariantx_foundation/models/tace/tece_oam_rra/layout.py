"""Packed feature views and shared adjoints for TECE-OAM-RRA."""

from functools import lru_cache, partial

import torch

from equivariantx.ops.common.channel_product import channel_product
from equivariantx.ops.common.cuda.execution import TiledProgram
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.frame_program import rotate
from equivariantx.utils.program import Program

from .expressions import packed_layout, paired_features
from .score import score_roles, tile_score


class LocalSplit(torch.autograd.Function):
    """Return packed views with a single shared gradient buffer."""

    @staticmethod
    def forward(ctx, features, ranges, metadata):
        ctx.kernel_metadata = metadata
        ctx.set_materialize_grads(False)
        return tuple(features.narrow(1, start, size) for start, size in ranges)

    @staticmethod
    def backward(ctx, *gradients):
        reference = next((g for g in gradients if g is not None), None)
        if reference is None:
            return None, None, None
        empty = reference.new_empty((reference.shape[0], 0))
        operands = [empty] + [g if g is not None else empty for g in gradients]
        mapping = tuple(range(len(operands)))
        program = tuple(
            (mapping, 0, 0, 1 << (i + 1))
            for i, g in enumerate(gradients)
            if g is not None
        )
        return (
            channel_product(ctx.kernel_metadata, repr(program), operands)[0],
            None,
            None,
        )


@lru_cache(maxsize=1)
def local_orders():
    """Return spherical row and radial group for each local component."""
    radial = [(m, l) for m in range(5) for l in range(m, 5)]
    return tuple(
        (l, l + sign * m, radial.index((m, l)))
        for m in range(5)
        for sign in ((1, -1) if m else (1,))
        for l in range(m, 5)
    )


@lru_cache(maxsize=32)
def paired_layout(nodes):
    """Identify an exact 64-channel endpoint rotation and its radial products."""
    program = Program(nodes)
    source, target, matrix = (
        program.input(0, "source", 1600),
        program.input(0, "target", 1600),
        program.input(5, "edge", 165),
    )
    if max(source, target, matrix) >= len(nodes):
        return None
    features = paired_features(program, source, target, 25, 64)
    pair = rotate(
        program, packed_layout(5, 4, 128), features, matrix, 25, 25, packed=True
    )
    if pair >= len(nodes):
        return None
    paths = tuple(
        ((i, group, i), 1.0) for i, (_, _, group) in enumerate(local_orders())
    )
    radial = (128, (25, 15, 25), paths)
    roles = {
        i: data[1]
        for i, (op, _, args, data) in enumerate(nodes)
        if op == "product" and len(args) == 3 and data[0] == radial
    }
    degrees = tuple(
        program.matmul(
            program.slice(matrix, l * (4 * l**2 - 1) // 3, d * d),
            program.slice(features, l * l * 128, d * 128),
            d,
            d,
            128,
        )
        for l in range(1, 5)
        for d in (2 * l + 1,)
    )
    return pair, roles, degrees, (matrix, source, target)


@lru_cache(maxsize=3)
def paired_source(weighted, by_degree=False):
    order = local_orders()
    cases = []
    for l in range(int(by_degree), 5):
        dim = 2 * l + 1
        offset = l * (4 * l * l - 1) // 3
        lines = [f"float v{k}=x[(node*25+{l * l + k})*64+c];" for k in range(dim)]
        rows = ((l, row, 0) for row in range(dim)) if by_degree else order
        for a, (degree, row, group) in enumerate(rows):
            if degree != l:
                continue
            if by_degree:
                dot = "0.f"
                for k in range(dim):
                    dot = f"fmaf(d[e*165+{offset + row * dim + k}],v{k},{dot})"
                lines.append(f"out{l}[(e*{dim}+{row})*128+t]={dot};")
                continue
            dot = "+".join(
                f"d[e*165+{offset + row * dim + k}]*v{k}" for k in range(dim)
            )
            w = f"*w[(e*15+{group})*128+t]" if weighted else ""
            lines.append(f"out[(e*25+{a})*128+t]=({dot}){w};")
        cases.append(f"case {l - int(by_degree)}:{{{''.join(lines)}break;}}")
    outputs = (
        ",".join(f"float* out{l}" for l in range(1, 5))
        if by_degree
        else "float* out,const float* w"
    )
    return f"""
    extern "C" __global__ void run(const float* x,const float* d,
        const long long* src,const long long* dst,{outputs}) {{
        long long e=blockIdx.x; int t=threadIdx.x, c=t%64;
        long long node=t<64?src[e]:dst[e];
        switch(blockIdx.y){{{"".join(cases)}}}
    }}"""


@lru_cache(maxsize=2)
def radial_source(sum_orders):
    cases = []
    for group in range(15):
        rows = [j for j, (_, _, g) in enumerate(local_orders()) if g == group]
        if sum_orders:
            dot = "+".join(f"a[(e*25+{j})*128+c]*b[(e*25+{j})*128+c]" for j in rows)
            lines = f"out[(e*15+{group})*128+c]={dot};"
        else:
            lines = f"float w=b[(e*15+{group})*128+c];" + "".join(
                f"out[(e*25+{j})*128+c]=a[(e*25+{j})*128+c]*w;" for j in rows
            )
        cases.append(f"case {group}:{{{lines}break;}}")
    return f"""
    extern "C" __global__ void run(const float* a,const float* b,float* out) {{
        long long e=blockIdx.x; int c=threadIdx.x;
        switch(blockIdx.y){{{"".join(cases)}}}
    }}"""


class PairedFrameTile(TiledProgram):
    """Gather both endpoints and rotate matching local-order features in one tile."""

    def __init__(self, *args, pair_layout=None, score_layout=None):
        super().__init__(*args)
        self.score_roles = score_layout or {}
        self.pair_layout = (
            pair_layout[:2]
            if pair_layout is not None
            and self.inputs[0].dtype == torch.float32
            and self.source.dtype == self.target.dtype == torch.int64
            else None
        )
        self.degrees = ()
        if self.pair_layout is not None:
            degrees, leaves = pair_layout[2:]
            if (
                all(
                    i in self.expressions and self.expressions[i][1] == leaves
                    for i in degrees
                )
                and self.inputs[0].is_cuda
                and self.inputs[5].dtype == torch.float32
                and all(
                    x.is_contiguous()
                    for x in (self.inputs[0], self.inputs[5], self.source, self.target)
                )
            ):
                self.degrees = degrees

    def expression_inputs(self, i):
        if self.degrees and i in self.expressions:
            leaves = self.expressions[i][1]
            if all(j in leaves for j in self.degrees):
                roots, self.degrees = self.degrees, ()
                if not any(j in self.values for j in roots):
                    rows = self.end - self.begin
                    x = self.inputs[0]
                    outputs = [x.new_empty((rows, self.nodes[j][1])) for j in roots]
                    if rows:
                        code = paired_source(False, by_degree=True)
                        operands = (
                            x,
                            self.inputs[5][self.begin : self.end],
                            self.source,
                            self.target,
                            *outputs,
                        )
                        runtime().launch(
                            [
                                (
                                    kernels((code,), x.device)[code],
                                    [v.data_ptr() for v in operands],
                                    rows,
                                    4,
                                    128,
                                    0,
                                )
                            ],
                            torch.cuda.current_stream(x.device).cuda_stream,
                        )
                    self.values.update(zip(roots, outputs))
                    for root in roots:
                        for child in self.dependencies["value", root]:
                            self.release(*child)
        return super().expression_inputs(i)

    def value(self, i, destination=None):
        if i in self.score_roles:
            return tile_score(self, i, destination)
        if destination is not None and (
            not destination.is_contiguous() or i in self.values
        ):
            return destination.copy_(self.value(i))
        if self.pair_layout is None:
            return super().value(i, destination)
        pair, roles = self.pair_layout
        if i != pair and i not in roles:
            return super().value(i, destination)
        if i in self.values:
            return self.values[i]
        _, size, args, _ = self.nodes[i]
        role = roles.get(i)
        rows = self.end - self.begin
        out = (
            destination
            if destination is not None
            else self.inputs[0].new_empty((rows, size))
        )
        if i == pair or (role == 2 and args[0] == pair and pair not in self.values):
            weights = self.value(args[1]).contiguous() if i != pair else None
            if weights is not None and weights.size(0) != rows:
                return super().value(i, destination)
            code = paired_source(weights is not None)
            operands = [
                self.inputs[0],
                self.inputs[5][self.begin : self.end],
                self.source,
                self.target,
                out,
            ]
            ptrs = [v.data_ptr() for v in operands] + [
                weights.data_ptr() if weights is not None else 0
            ]
            groups = 5
        else:
            a, b = (
                (args[2], args[1])
                if role == 0
                else (args[0], args[2])
                if role == 1
                else (args[0], args[1])
            )
            x, w = self.value(a).contiguous(), self.value(b).contiguous()
            if x.size(0) != rows or w.size(0) != rows:
                return super().value(i, destination)
            code = radial_source(role == 1)
            ptrs = [x.data_ptr(), w.data_ptr(), out.data_ptr()]
            groups = 15
        if rows:
            runtime().launch(
                [(kernels((code,), out.device)[code], ptrs, rows, groups, 128, 0)],
                torch.cuda.current_stream(out.device).cuda_stream,
            )
        for dep in self.dependencies["value", i]:
            self.release(*dep)
        self.values[i] = out
        return out


@lru_cache(maxsize=16)
def paired_tile(nodes):
    layout = paired_layout(nodes)
    score = score_roles(nodes)
    if layout is not None or score:
        return partial(PairedFrameTile, pair_layout=layout, score_layout=score)
    return TiledProgram
