"""Joint radial projections and permutation adjoints with TF32 rounding."""

import torch
import triton
import triton.language as tl

from .precision import require_tensors


@triton.jit
def project_columns(
    A,
    B,
    Columns,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    W: tl.constexpr,
    A0,
    A1: tl.constexpr,
    A2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    batch = tl.program_id(2).to(tl.int64)
    inner = tl.arange(0, BK)
    result = tl.zeros((BM, BN), tl.float32)
    for start in range(triton.cdiv(K, BK)):
        k = start * BK + inner
        a = tl.load(
            A + batch * A0 + row[:, None] * A1 + k[None, :] * A2,
            (row[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + batch * B0 + k[:, None] * B1 + column[None, :] * B2,
            (k[:, None] < K) & (column[None, :] < N),
            other=0,
        )
        # Round explicitly: Triton's implicit TF32 conversion may truncate.
        a = tl.inline_asm_elementwise(
            "cvt.rn.tf32.f32 $0, $1;",
            constraints="=f,f",
            args=[a],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        b = tl.inline_asm_elementwise(
            "cvt.rn.tf32.f32 $0, $1;",
            constraints="=f,f",
            args=[b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        result += tl.dot(a, b, input_precision="tf32")
    mapped = tl.load(Columns + batch * N + column, column < N, other=0)
    tl.store(
        C + batch * M * W + row[:, None] * W + mapped[None, :],
        result,
        (row[:, None] < M) & (column[None, :] < N),
    )


@torch.library.custom_op("eqx_oam_xl::projected_heads", mutates_args=())
def projected_heads(
    features: list[torch.Tensor],
    weights: list[torch.Tensor],
    columns: list[torch.Tensor],
    width: int,
) -> torch.Tensor:
    """Project heads into disjoint columns covering the complete path width."""
    require_tensors((*features, *weights))
    first = features[0]
    batch, rows = first.shape[:2]
    output = first.new_empty((batch, rows, width))
    if not rows:
        return output
    for left, right, index in zip(features, weights, columns):
        index = index.contiguous()
        if (
            first.is_cuda
            and first.dtype == torch.float32
            and torch.backends.cuda.matmul.fp32_precision == "tf32"
        ):
            size, inner = right.shape[-1], left.shape[-1]
            project_columns[(triton.cdiv(rows, 32), triton.cdiv(size, 128), batch)](
                left,
                right,
                index,
                output,
                rows,
                size,
                inner,
                width,
                *left.stride(),
                *right.stride(),
                32,
                128,
                32,
                num_warps=4,
                num_stages=3,
            )
        else:
            output.scatter_(2, index[:, None].expand(batch, rows, -1), left @ right)
    return output


@projected_heads.register_fake
def fake_projected_heads(features, weights, columns, width):
    return features[0].new_empty((*features[0].shape[:2], width))


def setup_context(ctx, inputs, output):
    features, weights, columns, width = inputs
    ctx.count = len(features)
    ctx.save_for_backward(*features, *weights, *columns)


def backward(ctx, gradient):
    n = ctx.count
    features = ctx.saved_tensors[:n]
    weights = ctx.saved_tensors[n : 2 * n]
    columns = ctx.saved_tensors[2 * n :]
    inputs, parameters = [], []
    for left, right, selected in zip(
        features, weights, split_columns(gradient, list(columns))
    ):
        inputs.append(selected @ right.transpose(-1, -2))
        parameters.append(left.transpose(-1, -2) @ selected)
    return inputs, parameters, [None] * n, None


projected_heads.register_autograd(backward, setup_context=setup_context)


@triton.jit
def columns_kernel(
    A,
    Columns,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    W: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    JOIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    row, column = index // N, index % N
    mapped = tl.load(Columns + batch * N + column, index < M * N, other=0)
    if JOIN:
        address = batch * A0 + row * A1 + column * A2
        destination = batch * M * W + row * W + mapped
    else:
        address = batch * A0 + row * A1 + mapped * A2
        destination = batch * M * N + index
    value = tl.load(A + address, index < M * N, other=0)
    tl.store(C + destination, value, index < M * N)


@torch.library.custom_op("eqx_oam_xl::split_columns", mutates_args=())
def split_columns(
    value: torch.Tensor, columns: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Gather the complete column partition with one inverse adjoint."""
    if not value.is_cuda or not value.shape[1]:
        return [
            value.gather(2, index[:, None].expand(-1, value.shape[1], -1))
            for index in columns
        ]
    output = []
    batch, rows, width = value.shape
    for index in columns:
        index = index.contiguous()
        size = index.shape[-1]
        result = value.new_empty((batch, rows, size))
        columns_kernel[(triton.cdiv(rows * size, 1024), batch)](
            value,
            index,
            result,
            rows,
            size,
            width,
            *value.stride(),
            False,
            1024,
        )
        output.append(result)
    return output


@split_columns.register_fake
def fake_split_columns(value, columns):
    return [value.new_empty((*value.shape[:2], index.shape[-1])) for index in columns]


def setup_split(ctx, inputs, output):
    value, columns = inputs
    ctx.save_for_backward(*columns)
    ctx.width = value.shape[-1]
    ctx.shapes = [tensor.shape for tensor in output]
    ctx.device, ctx.dtype = value.device, value.dtype


def split_backward(ctx, gradients):
    gradients = [
        torch.zeros(shape, device=ctx.device, dtype=ctx.dtype) if grad is None else grad
        for shape, grad in zip(ctx.shapes, gradients)
    ]
    return join_columns(gradients, list(ctx.saved_tensors), ctx.width), [None] * len(
        gradients
    )


split_columns.register_autograd(split_backward, setup_context=setup_split)


@torch.library.custom_op("eqx_oam_xl::join_columns", mutates_args=())
def join_columns(
    values: list[torch.Tensor], columns: list[torch.Tensor], width: int
) -> torch.Tensor:
    """Write disjoint blocks into the complete column partition."""
    first = values[0]
    batch, rows = first.shape[:2]
    output = first.new_empty((batch, rows, width))
    if not rows:
        return output
    for value, index in zip(values, columns):
        index = index.contiguous()
        if first.is_cuda:
            size = index.shape[-1]
            columns_kernel[(triton.cdiv(rows * size, 1024), batch)](
                value,
                index,
                output,
                rows,
                size,
                width,
                *value.stride(),
                True,
                1024,
            )
        else:
            output.scatter_(2, index[:, None].expand(-1, rows, -1), value)
    return output


@join_columns.register_fake
def fake_join_columns(values, columns, width):
    return values[0].new_empty((*values[0].shape[:2], width))


def setup_join(ctx, inputs, output):
    ctx.save_for_backward(*inputs[1])


def join_backward(ctx, gradient):
    columns = list(ctx.saved_tensors)
    return split_columns(gradient, columns), [None] * len(columns), None


join_columns.register_autograd(join_backward, setup_context=setup_join)


def project_degree_jets(networks, radial, jet):
    """Project compatible layer batches into their original convolution columns."""
    from .radial_jet import AttachRadialJet, packed_hidden_jets

    groups = {}
    for i, net in enumerate(networks):
        key = (
            net.out_features,
            tuple(net.active),
            tuple((head.mlp[-1].in_dim, head.mlp[-1].out_dim) for head in net.branches),
        )
        groups.setdefault(key, []).append(i)
    values = {}
    parts = radial.split(networks[0].in_features, -1)
    slopes = jet[1].split(networks[0].in_features, -1)
    for degree, part in enumerate(parts):
        ids = [i for i, net in enumerate(networks) if degree in net.active]
        if not ids:
            continue
        heads = [networks[i].branches[networks[i].active.index(degree)] for i in ids]
        value, tangent = packed_hidden_jets(
            heads, part.detach(), slopes[degree], batched=True
        )
        values[degree] = ids, AttachRadialJet.apply(value, tangent, jet[0])
    result = [None] * len(networks)
    for (width, active, _), ids in groups.items():
        inputs, matrices, columns = [], [], []
        offset = 0
        for degree in active:
            positions, hidden = values[degree]
            selected = [positions.index(i) for i in ids]
            if selected == list(range(selected[0], selected[-1] + 1)):
                hidden = hidden[selected[0] : selected[-1] + 1]
            else:
                hidden = hidden[selected]
            heads = [
                networks[i].branches[networks[i].active.index(degree)].mlp[-1]
                for i in ids
            ]
            size = heads[0].out_dim
            inputs.append(hidden)
            matrices.append(torch.stack([head.get_weight() for head in heads]))
            indices = [
                networks[i].inverse_order[offset : offset + size]
                if networks[i].output_order is not None
                else torch.arange(offset, offset + size, device=hidden.device)
                for i in ids
            ]
            columns.append(torch.stack(indices))
            offset += size
        assert offset == width
        output = projected_heads(inputs, matrices, columns, width)
        for i, features in zip(ids, output.unbind(0)):
            result[i] = features, features.new_empty((0, width))
    return result
