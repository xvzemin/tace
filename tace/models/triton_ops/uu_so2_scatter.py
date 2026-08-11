################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
import triton
import triton.language as tl


@triton.jit
def _uu_so2_scatter_forward(
    x_ptr,
    weight_ptr,
    source_ptr,
    target_ptr,
    wigner_ptr,
    wigner_inv_ptr,
    path_out_ptr,
    path_in_ptr,
    path_weight_index_ptr,
    path_weight_ptr,
    output_ptr,
    num_edges: tl.constexpr,
    num_channels: tl.constexpr,
    num_input_components: tl.constexpr,
    num_so2_components: tl.constexpr,
    num_output_components: tl.constexpr,
    num_paths: tl.constexpr,
    num_weights: tl.constexpr,
    block_channels: tl.constexpr,
    block_outputs: tl.constexpr,
    use_fp64: tl.constexpr,
):
    edge = tl.program_id(0)
    output_block = tl.program_id(1)
    channel_block = tl.program_id(2)

    channels = channel_block * block_channels + tl.arange(0, block_channels)
    outputs = output_block * block_outputs + tl.arange(0, block_outputs)
    channel_mask = channels < num_channels
    output_mask = outputs < num_output_components

    source = tl.load(source_ptr + edge).to(tl.int64)
    target = tl.load(target_ptr + edge).to(tl.int64)
    if use_fp64:
        accumulator = tl.zeros(
            (block_outputs, block_channels),
            dtype=tl.float64,
        )
    else:
        accumulator = tl.zeros(
            (block_outputs, block_channels),
            dtype=tl.float32,
        )

    input_components = tl.arange(0, triton.next_power_of_2(num_input_components))
    input_mask = input_components < num_input_components
    for path in range(num_paths):
        path_out = tl.load(path_out_ptr + path).to(tl.int64)
        path_in = tl.load(path_in_ptr + path).to(tl.int64)
        weight_index = tl.load(path_weight_index_ptr + path).to(tl.int64)
        path_scale = tl.load(path_weight_ptr + path)

        node_values = tl.load(
            x_ptr
            + source * num_input_components * num_channels
            + input_components[:, None] * num_channels
            + channels[None, :],
            mask=input_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        rotation = tl.load(
            wigner_ptr
            + edge * num_so2_components * num_input_components
            + path_in * num_input_components
            + input_components,
            mask=input_mask,
            other=0.0,
        )
        rotated = tl.sum(node_values * rotation[:, None], axis=0)
        edge_weight = tl.load(
            weight_ptr
            + edge * num_weights * num_channels
            + weight_index * num_channels
            + channels,
            mask=channel_mask,
            other=0.0,
        )
        inverse_rotation = tl.load(
            wigner_inv_ptr
            + edge * num_output_components * num_so2_components
            + outputs * num_so2_components
            + path_out,
            mask=output_mask,
            other=0.0,
        )
        accumulator += (
            inverse_rotation[:, None]
            * rotated[None, :]
            * edge_weight[None, :]
            * path_scale
        )

    tl.atomic_add(
        output_ptr
        + target * num_output_components * num_channels
        + outputs[:, None] * num_channels
        + channels[None, :],
        accumulator,
        mask=output_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _uu_so2_linear_forward(
    rotated_ptr,
    weight_ptr,
    path_out_ptr,
    path_in_ptr,
    path_weight_index_ptr,
    path_weight_ptr,
    mixed_ptr,
    num_channels: tl.constexpr,
    num_so2_components: tl.constexpr,
    num_paths: tl.constexpr,
    num_weights: tl.constexpr,
    block_channels: tl.constexpr,
):
    edge = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * block_channels + tl.arange(0, block_channels)
    channel_mask = channels < num_channels

    for path in range(num_paths):
        path_out = tl.load(path_out_ptr + path).to(tl.int64)
        path_in = tl.load(path_in_ptr + path).to(tl.int64)
        weight_index = tl.load(path_weight_index_ptr + path).to(tl.int64)
        path_scale = tl.load(path_weight_ptr + path)
        rotated = tl.load(
            rotated_ptr
            + edge * num_so2_components * num_channels
            + path_in * num_channels
            + channels,
            mask=channel_mask,
            other=0.0,
        )
        edge_weight = tl.load(
            weight_ptr
            + edge * num_weights * num_channels
            + weight_index * num_channels
            + channels,
            mask=channel_mask,
            other=0.0,
        )
        tl.atomic_add(
            mixed_ptr
            + edge * num_so2_components * num_channels
            + path_out * num_channels
            + channels,
            rotated * edge_weight * path_scale,
            mask=channel_mask,
        )


@triton.jit
def _uu_so2_linear_backward(
    grad_mixed_ptr,
    rotated_ptr,
    weight_ptr,
    path_out_ptr,
    path_in_ptr,
    path_weight_index_ptr,
    path_weight_ptr,
    grad_rotated_ptr,
    grad_weight_ptr,
    num_channels: tl.constexpr,
    num_so2_components: tl.constexpr,
    num_paths: tl.constexpr,
    num_weights: tl.constexpr,
    block_channels: tl.constexpr,
):
    edge = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * block_channels + tl.arange(0, block_channels)
    channel_mask = channels < num_channels

    for path in range(num_paths):
        path_out = tl.load(path_out_ptr + path).to(tl.int64)
        path_in = tl.load(path_in_ptr + path).to(tl.int64)
        weight_index = tl.load(path_weight_index_ptr + path).to(tl.int64)
        path_scale = tl.load(path_weight_ptr + path)
        grad_mixed = tl.load(
            grad_mixed_ptr
            + edge * num_so2_components * num_channels
            + path_out * num_channels
            + channels,
            mask=channel_mask,
            other=0.0,
        )
        rotated = tl.load(
            rotated_ptr
            + edge * num_so2_components * num_channels
            + path_in * num_channels
            + channels,
            mask=channel_mask,
            other=0.0,
        )
        edge_weight = tl.load(
            weight_ptr
            + edge * num_weights * num_channels
            + weight_index * num_channels
            + channels,
            mask=channel_mask,
            other=0.0,
        )
        scaled_grad = grad_mixed * path_scale
        tl.atomic_add(
            grad_rotated_ptr
            + edge * num_so2_components * num_channels
            + path_in * num_channels
            + channels,
            scaled_grad * edge_weight,
            mask=channel_mask,
        )
        tl.atomic_add(
            grad_weight_ptr
            + edge * num_weights * num_channels
            + weight_index * num_channels
            + channels,
            scaled_grad * rotated,
            mask=channel_mask,
        )


def _streaming_uu_so2_scatter_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    wigner: torch.Tensor,
    wigner_inv: torch.Tensor,
    path_out: torch.Tensor,
    path_in: torch.Tensor,
    path_weight_index: torch.Tensor,
    path_weight: torch.Tensor,
    num_weights: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    weight = weight.contiguous()
    wigner = wigner.contiguous()
    wigner_inv = wigner_inv.contiguous()
    grad_output = grad_output.contiguous()
    num_edges = source.numel()
    num_channels = x.size(2)
    num_so2_components = wigner.size(1)
    block_channels = min(32, triton.next_power_of_2(num_channels))
    grid = (
        num_edges,
        triton.cdiv(num_channels, block_channels),
    )

    source_values = x.index_select(0, source)
    rotated = torch.bmm(wigner, source_values)
    mixed = torch.zeros_like(rotated)
    _uu_so2_linear_forward[grid](
        rotated,
        weight.view(num_edges, num_weights, num_channels),
        path_out,
        path_in,
        path_weight_index,
        path_weight,
        mixed,
        num_channels=num_channels,
        num_so2_components=num_so2_components,
        num_paths=path_out.numel(),
        num_weights=num_weights,
        block_channels=block_channels,
    )

    edge_grad = grad_output.index_select(0, target)
    grad_mixed = torch.bmm(wigner_inv.transpose(1, 2), edge_grad)
    grad_wigner_inv = torch.bmm(edge_grad, mixed.transpose(1, 2))
    del mixed, edge_grad

    grad_rotated = torch.zeros_like(rotated)
    grad_weight = torch.zeros(
        num_edges,
        num_weights,
        num_channels,
        dtype=weight.dtype,
        device=weight.device,
    )
    _uu_so2_linear_backward[grid](
        grad_mixed,
        rotated,
        weight.view(num_edges, num_weights, num_channels),
        path_out,
        path_in,
        path_weight_index,
        path_weight,
        grad_rotated,
        grad_weight,
        num_channels=num_channels,
        num_so2_components=num_so2_components,
        num_paths=path_out.numel(),
        num_weights=num_weights,
        block_channels=block_channels,
    )
    del grad_mixed, rotated

    grad_wigner = torch.bmm(
        grad_rotated,
        source_values.transpose(1, 2),
    )
    source_grad = torch.bmm(
        wigner.transpose(1, 2),
        grad_rotated,
    )
    grad_x = torch.zeros_like(x)
    grad_x.index_add_(0, source, source_grad)
    return grad_x, grad_weight.view_as(weight), grad_wigner, grad_wigner_inv


def _streaming_uu_so2_scatter_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    wigner: torch.Tensor,
    wigner_inv: torch.Tensor,
    path_out: torch.Tensor,
    path_in: torch.Tensor,
    path_weight_index: torch.Tensor,
    path_weight: torch.Tensor,
    num_weights: int,
) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    wigner = wigner.contiguous()
    wigner_inv = wigner_inv.contiguous()
    num_edges = source.numel()
    num_channels = x.size(2)
    num_input_components = x.size(1)
    num_so2_components = wigner.size(1)
    num_output_components = wigner_inv.size(1)
    output = x.new_zeros(
        x.size(0),
        num_output_components,
        num_channels,
    )
    block_channels = min(32, triton.next_power_of_2(num_channels))
    block_outputs = min(
        4,
        triton.next_power_of_2(num_output_components),
    )
    grid = (
        num_edges,
        triton.cdiv(num_output_components, block_outputs),
        triton.cdiv(num_channels, block_channels),
    )
    _uu_so2_scatter_forward[grid](
        x,
        weight.view(num_edges, num_weights, num_channels),
        source,
        target,
        wigner,
        wigner_inv,
        path_out,
        path_in,
        path_weight_index,
        path_weight,
        output,
        num_edges=num_edges,
        num_channels=num_channels,
        num_input_components=num_input_components,
        num_so2_components=num_so2_components,
        num_output_components=num_output_components,
        num_paths=path_out.numel(),
        num_weights=num_weights,
        block_channels=block_channels,
        block_outputs=block_outputs,
        use_fp64=x.dtype == torch.float64,
    )
    return output


def _check_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    wigner: torch.Tensor,
    wigner_inv: torch.Tensor,
) -> None:
    if not x.is_cuda:
        raise ValueError("TACE_USE_EQX=1 requires CUDA tensors")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "The Triton uuSO2 scatter operator supports float32 and float64"
        )
    tensors = (weight, source, target, wigner, wigner_inv)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("All Triton uuSO2 scatter inputs must share a device")
    if weight.dim() != 2 or weight.size(0) != source.numel():
        raise ValueError("Triton uuSO2 scatter requires one weight vector per edge")


class _UUSO2ScatterBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output,
        x,
        weight,
        source,
        target,
        wigner,
        wigner_inv,
        path_out,
        path_in,
        path_weight_index,
        path_weight,
        num_weights,
    ):
        grads = _streaming_uu_so2_scatter_backward(
            grad_output,
            x,
            weight,
            source,
            target,
            wigner,
            wigner_inv,
            path_out,
            path_in,
            path_weight_index,
            path_weight,
            num_weights,
        )
        ctx.save_for_backward(
            grad_output,
            x,
            weight,
            source,
            target,
            wigner,
            wigner_inv,
            path_out,
            path_in,
            path_weight_index,
            path_weight,
        )
        ctx.num_weights = num_weights
        return grads

    @staticmethod
    def backward(
        ctx,
        grad_grad_x,
        grad_grad_weight,
        grad_grad_wigner,
        grad_grad_wigner_inv,
    ):
        (
            grad_output,
            x,
            weight,
            source,
            target,
            wigner,
            wigner_inv,
            path_out,
            path_in,
            path_weight_index,
            path_weight,
        ) = ctx.saved_tensors
        num_weights = ctx.num_weights

        def add(current, value):
            return value if current is None else current + value

        grad_grad_output = None
        grad_x = None
        grad_weight = None
        grad_wigner = None
        grad_wigner_inv = None

        # The operator is quadrilinear, so each double-backward term can
        # reuse the fused forward and first-backward implementations.
        if grad_grad_x is not None:
            grad_grad_output = add(
                grad_grad_output,
                _streaming_uu_so2_scatter_forward(
                    grad_grad_x,
                    weight,
                    source,
                    target,
                    wigner,
                    wigner_inv,
                    path_out,
                    path_in,
                    path_weight_index,
                    path_weight,
                    num_weights,
                ),
            )
            _, dw, d_wigner, d_wigner_inv = _streaming_uu_so2_scatter_backward(
                grad_output,
                grad_grad_x,
                weight,
                source,
                target,
                wigner,
                wigner_inv,
                path_out,
                path_in,
                path_weight_index,
                path_weight,
                num_weights,
            )
            grad_weight = add(grad_weight, dw)
            grad_wigner = add(grad_wigner, d_wigner)
            grad_wigner_inv = add(grad_wigner_inv, d_wigner_inv)

        if grad_grad_weight is not None:
            grad_grad_output = add(
                grad_grad_output,
                _streaming_uu_so2_scatter_forward(
                    x,
                    grad_grad_weight,
                    source,
                    target,
                    wigner,
                    wigner_inv,
                    path_out,
                    path_in,
                    path_weight_index,
                    path_weight,
                    num_weights,
                ),
            )
            dx, _, d_wigner, d_wigner_inv = _streaming_uu_so2_scatter_backward(
                grad_output,
                x,
                grad_grad_weight,
                source,
                target,
                wigner,
                wigner_inv,
                path_out,
                path_in,
                path_weight_index,
                path_weight,
                num_weights,
            )
            grad_x = add(grad_x, dx)
            grad_wigner = add(grad_wigner, d_wigner)
            grad_wigner_inv = add(grad_wigner_inv, d_wigner_inv)

        if grad_grad_wigner is not None:
            grad_grad_output = add(
                grad_grad_output,
                _streaming_uu_so2_scatter_forward(
                    x,
                    weight,
                    source,
                    target,
                    grad_grad_wigner,
                    wigner_inv,
                    path_out,
                    path_in,
                    path_weight_index,
                    path_weight,
                    num_weights,
                ),
            )
            dx, dw, _, d_wigner_inv = _streaming_uu_so2_scatter_backward(
                grad_output,
                x,
                weight,
                source,
                target,
                grad_grad_wigner,
                wigner_inv,
                path_out,
                path_in,
                path_weight_index,
                path_weight,
                num_weights,
            )
            grad_x = add(grad_x, dx)
            grad_weight = add(grad_weight, dw)
            grad_wigner_inv = add(grad_wigner_inv, d_wigner_inv)

        if grad_grad_wigner_inv is not None:
            grad_grad_output = add(
                grad_grad_output,
                _streaming_uu_so2_scatter_forward(
                    x,
                    weight,
                    source,
                    target,
                    wigner,
                    grad_grad_wigner_inv,
                    path_out,
                    path_in,
                    path_weight_index,
                    path_weight,
                    num_weights,
                ),
            )
            dx, dw, d_wigner, _ = _streaming_uu_so2_scatter_backward(
                grad_output,
                x,
                weight,
                source,
                target,
                wigner,
                grad_grad_wigner_inv,
                path_out,
                path_in,
                path_weight_index,
                path_weight,
                num_weights,
            )
            grad_x = add(grad_x, dx)
            grad_weight = add(grad_weight, dw)
            grad_wigner = add(grad_wigner, d_wigner)

        return (
            grad_grad_output,
            grad_x,
            grad_weight,
            None,
            None,
            grad_wigner,
            grad_wigner_inv,
            None,
            None,
            None,
            None,
            None,
        )


class _UUSO2ScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        source,
        target,
        wigner,
        wigner_inv,
        path_out,
        path_in,
        path_weight_index,
        path_weight,
        num_weights,
    ):
        _check_inputs(x, weight, source, target, wigner, wigner_inv)
        source = source.contiguous()
        target = target.contiguous()
        path_out = path_out.contiguous()
        path_in = path_in.contiguous()
        path_weight_index = path_weight_index.contiguous()
        path_weight = path_weight.to(dtype=x.dtype).contiguous()
        output = _streaming_uu_so2_scatter_forward(
            x,
            weight,
            source,
            target,
            wigner,
            wigner_inv,
            path_out,
            path_in,
            path_weight_index,
            path_weight,
            num_weights,
        )
        ctx.save_for_backward(
            x,
            weight,
            source,
            target,
            wigner,
            wigner_inv,
            path_out,
            path_in,
            path_weight_index,
            path_weight,
        )
        ctx.num_weights = num_weights
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            weight,
            source,
            target,
            wigner,
            wigner_inv,
            path_out,
            path_in,
            path_weight_index,
            path_weight,
        ) = ctx.saved_tensors
        grad_x, grad_weight, grad_wigner, grad_wigner_inv = (
            _UUSO2ScatterBackwardFunction.apply(
                grad_output,
                x,
                weight,
                source,
                target,
                wigner,
                wigner_inv,
                path_out,
                path_in,
                path_weight_index,
                path_weight,
                ctx.num_weights,
            )
        )

        return (
            grad_x if ctx.needs_input_grad[0] else None,
            grad_weight if ctx.needs_input_grad[1] else None,
            None,
            None,
            grad_wigner if ctx.needs_input_grad[4] else None,
            grad_wigner_inv if ctx.needs_input_grad[5] else None,
            None,
            None,
            None,
            None,
            None,
        )


def uu_so2_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    edge_index: torch.Tensor,
    wigner: torch.Tensor,
    wigner_inv: torch.Tensor,
    path_out: torch.Tensor,
    path_in: torch.Tensor,
    path_weight_index: torch.Tensor,
    path_weight: torch.Tensor,
    num_weights: int,
) -> torch.Tensor:
    if edge_index.size(1) == 0:
        zero = (x.sum() + weight.sum() + wigner.sum() + wigner_inv.sum()) * 0.0
        return (
            x.new_zeros(
                x.size(0),
                wigner_inv.size(1),
                x.size(2),
            )
            + zero
        )
    return _UUSO2ScatterFunction.apply(
        x,
        weight,
        edge_index[0],
        edge_index[1],
        wigner,
        wigner_inv,
        path_out,
        path_in,
        path_weight_index,
        path_weight,
        num_weights,
    )


class UUSO2Scatter(torch.nn.Module):
    def __init__(self, linear: torch.nn.Module) -> None:
        super().__init__()
        self.num_weights = linear.num_weights
        self.register_buffer(
            "path_out",
            linear.path_out.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "path_in",
            linear.path_in.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "path_weight_index",
            linear.path_weight_index.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "path_weight",
            linear.path_weight.detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        return uu_so2_scatter(
            x,
            weight,
            edge_index,
            wigner,
            wigner_inv,
            self.path_out,
            self.path_in,
            self.path_weight_index,
            self.path_weight,
            self.num_weights,
        )
