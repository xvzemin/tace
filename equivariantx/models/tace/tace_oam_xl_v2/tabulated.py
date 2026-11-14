"""Grouped differentiable table contractions and their mixed adjoints."""

import torch

from equivariantx.ops.common.cuda.pointwise import evaluate as pointwise


@torch.library.custom_op("eqx_oam_xl::table_contract", mutates_args=())
def table_contract(
    role: int,
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    ptr: torch.Tensor,
    tasks: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one output of ``sum_group a[e,k] * b[group,k,c] * y[e,c]``."""
    values = (a, b, y)
    if role not in (0, 1, 2):
        raise ValueError("Expected a bilinear role")
    out = torch.empty_like(values[role], memory_format=torch.contiguous_format)
    if not a.shape[0]:
        return out.zero_()
    if a.device.type == "cpu":
        positions = ptr.tolist()
        for group, (begin, end) in enumerate(zip(positions, positions[1:])):
            if role == 2:
                out[begin:end] = a[begin:end] @ b[group]
            elif role == 0:
                out[begin:end] = y[begin:end] @ b[group].T
            else:
                out[group] = a[begin:end].T @ y[begin:end]
        return out
    a, b, y, ptr = (v.contiguous() for v in (a, b, y, ptr))
    k, c = b.shape[1:]
    if role == 1:
        body = f"""
        I group=i/({k}*{c}), row=(i/{c})%{k}, col=i%{c};
        T total=0;
        for(I e=p3[group];e<p3[group+1];++e)
            total += p0[e*{k}+row]*p2[e*{c}+col];
        out[i]=total;
        """
    else:
        columns = c if role == 2 else k
        product = (
            f"p0[row*{k}+q]*p1[(lower*{k}+q)*{c}+col]"
            if role == 2
            else f"p2[row*{c}+q]*p1[(lower*{k}+col)*{c}+q]"
        )
        body = f"""
        I row=i/{columns}, col=i%{columns};
        I lower=0, upper={b.shape[0]};
        while(lower<upper) {{
            I middle=(lower+upper)/2;
            if(p3[middle+1]<=row) lower=middle+1; else upper=middle;
        }}
        T total=0;
        for(int q=0;q<{k if role == 2 else c};++q) total += {product};
        out[i]=total;
        """
    return pointwise(out, (a, b, y, ptr), body)


@table_contract.register_fake
def fake_table(role, a, b, y, ptr, tasks):
    return torch.empty_like((a, b, y)[role], memory_format=torch.contiguous_format)


def table_context(ctx, inputs, output):
    role, a, b, y, ptr, tasks = inputs
    ctx.role = role
    ctx.save_for_backward(a, b, y, ptr, tasks)


def table_backward(ctx, gradient):
    a, b, y, ptr, tasks = ctx.saved_tensors
    values = [a, b, y]
    values[ctx.role] = gradient.contiguous()
    gradients = [None, None, None]
    for role in range(3):
        if role != ctx.role and ctx.needs_input_grad[role + 1]:
            gradients[role] = table_contract(role, *values, ptr, tasks)
    return (None, *gradients, None, None)


table_contract.register_autograd(table_backward, setup_context=table_context)


def interpolate_table(weights, table, ptr, tasks):
    """Apply per-group cardinal weights to differentiable network samples."""
    # The inactive operand carries only its output shape; it is never read.
    shape = weights.new_empty(1).expand(weights.shape[0], table.shape[-1])
    return table_contract(
        2, weights.contiguous(), table.contiguous(), shape, ptr, tasks
    )


def table_tasks(ptr, block=32):
    """Make static launch tiles from CPU group lengths."""
    positions = ptr.tolist()
    tasks = [
        (group, start)
        for group, (begin, end) in enumerate(zip(positions, positions[1:]))
        for start in range(0, end - begin, block)
    ]
    return torch.tensor(tasks, dtype=torch.int64).reshape(-1, 2)
