"""Joint radial projections and their column-permutation adjoints."""

import torch

from equivariantx.ops.common.cuda.pointwise import launch, scalar_type

from .precision import require_tensors


def permute_projected_columns(output, columns):
    """Permute each projected row in place on the current CUDA stream."""
    batch, rows, width = output.shape
    order = torch.cat(columns, dim=1).long()
    row_bytes = width * output.element_size()
    blocks = min(batch * rows, 65535)
    args = (output, order)
    if row_bytes <= 48 * 1024:
        shared, pointer, storage = row_bytes, "", "extern __shared__ T tile[];"
    else:
        # Bound scratch independently of the number of edges.
        blocks = min(blocks, max(1, (16 << 20) // row_bytes))
        args += (output.new_empty((blocks, width)),)
        shared, pointer = 0, "T* workspace,"
        storage = "T* tile = workspace + I(blockIdx.x) * width;"
    source = f"""
    using T = {scalar_type(output.dtype)};
    using I = long long;
    extern "C" __global__ void run(
        T* output, const I* columns, {pointer} I rows, I width, I count
    ) {{
        {storage}
        for (I r = blockIdx.x; r < count; r += gridDim.x) {{
            T* row = output + r * width;
            const I* order = columns + (r / rows) * width;
            for (I j = threadIdx.x; j < width; j += blockDim.x)
                tile[order[j]] = row[j];
            __syncthreads();
            for (I j = threadIdx.x; j < width; j += blockDim.x)
                row[j] = tile[j];
            __syncthreads();
        }}
    }}
    """
    launch(
        source, args, (rows, width, batch * rows),
        blocks * 128, threads=128, shared=shared,
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
    if not output.numel():
        return output
    if first.is_cuda and torch.version.hip is None:
        offset = 0
        for left, right in zip(features, weights):
            size = right.shape[-1]
            torch.bmm(left, right, out=output.narrow(-1, offset, size))
            offset += size
        permute_projected_columns(output, columns)
    else:
        for left, right, index in zip(features, weights, columns):
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


@torch.library.custom_op("eqx_oam_xl::split_columns", mutates_args=())
def split_columns(
    value: torch.Tensor, columns: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Gather the complete column partition with one inverse adjoint."""
    return [
        value.gather(2, index[:, None].expand(-1, value.shape[1], -1))
        for index in columns
    ]


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
    for value, index in zip(values, columns):
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
