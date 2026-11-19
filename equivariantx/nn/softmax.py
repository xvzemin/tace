"""Segmented softmax."""

from math import prod

import torch

from ..ops import get_kernel, resolve_backend
from ..ops.precision import validate_precision


def graph_softmax(
    scores, target, num_nodes, weight=None, eps=1e-16, *, backend=None, fused=None
):
    """Apply a weighted softmax to scalar logits within each target group.

    Parameters
    ----------
    scores : torch.Tensor
        Logits of shape ``(items, heads)``.
    target : torch.Tensor
        Integer group indices of shape ``(items,)``.
    num_nodes : int
        Number of groups, including empty groups.
    weight : torch.Tensor, optional
        Nonnegative weights of shape ``(items, 1)`` or ``(items, heads)``.
    eps : float, optional
        Constant added to the denominator after subtracting the group maximum.
    backend : {None, 'torch', 'cuda'}, optional
        Execution backend. CPU inputs use PyTorch.
    fused : bool, optional
        Select CUDA if true, PyTorch otherwise.

    Returns
    -------
    torch.Tensor
        Normalized weights of shape ``(items, heads)``.

    Examples
    --------
    >>> from equivariantx.nn import graph_softmax
    >>> graph_softmax(torch.zeros(3, 1), torch.tensor([0, 0, 1]), 2).flatten()
    tensor([0.5000, 0.5000, 1.0000])
    """
    backend = resolve_backend(backend, scores.device)
    validate_precision((scores, weight))
    if fused is not None:
        backend = "cuda" if fused else "torch"
    if scores.ndim != 2:
        raise ValueError("Scores must have shape (items, heads).")
    if weight is not None and (
        weight.ndim != 2
        or weight.shape[0] != scores.shape[0]
        or weight.shape[1] not in (1, scores.shape[1])
    ):
        raise ValueError("Weights must have shape (items, 1) or (items, heads).")
    if weight is not None and (
        weight.device != scores.device or weight.dtype != scores.dtype
    ):
        raise ValueError("Scores and weights must have the same device and dtype.")
    if (
        target.shape != scores.shape[:1]
        or target.device != scores.device
        or target.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Group indices must match the items and device of scores.")
    target = target.long()
    if backend == "cuda" and scores.is_cuda:
        from ..ops.common.attention import normalized_exp

        normalizer = (
            scores.new_ones((scores.shape[0], 1))
            if weight is None
            else weight.contiguous()
        )
        values = normalized_exp(scores.contiguous(), normalizer, target, num_nodes, eps)
    else:
        values = get_kernel("graph_normalized_exp", "torch")(
            scores, weight, target, num_nodes, eps
        )
    return values if weight is None else values * weight


class GraphSoftmax(torch.nn.Module):
    """Apply a segmented softmax along one dimension.

    Parameters
    ----------
    eps : float, optional
        Constant added to each shifted denominator.
    backend : {None, 'torch', 'cuda'}, optional
        Execution backend.

    Examples
    --------
    >>> from equivariantx import nn
    >>> nn.GraphSoftmax()(torch.zeros(3), ptr=torch.tensor([0, 2, 3]))
    tensor([0.5000, 0.5000, 1.0000])
    """

    def __init__(self, eps=1e-16, *, backend=None):
        super().__init__()
        self.eps = eps
        self.backend = resolve_backend(backend)

    def forward(
        self, src, index=None, ptr=None, num_nodes=None, dim=0, exp_rescale=None
    ):
        """Normalize ``src`` over groups along ``dim``.

        Parameters
        ----------
        src : torch.Tensor
            Input logits.
        index : torch.Tensor, optional
            Group index for each item along ``dim``.
        ptr : torch.Tensor, optional
            CSR offsets for contiguous groups, in place of ``index``.
        num_nodes : int, optional
            Number of groups. Inferred from ``ptr`` or ``index`` if omitted.
        dim : int, optional
            Grouped dimension. Defaults to zero.
        exp_rescale : torch.Tensor, optional
            Nonnegative weights broadcastable to ``src``.

        Returns
        -------
        torch.Tensor
            Normalized weights of the same shape as ``src``.
        """
        values = src.movedim(dim, 0)
        items = values.shape[0]
        if ptr is not None:
            if (
                ptr.ndim != 1
                or ptr.numel() < 1
                or ptr.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("ptr must be a nonempty integer vector.")
            num_nodes = ptr.numel() - 1
            index = torch.arange(num_nodes, device=ptr.device).repeat_interleave(
                ptr.diff().long(), output_size=items
            )
        elif index is None:
            raise ValueError("Pass index or ptr.")
        elif num_nodes is None:
            num_nodes = int(index.max()) + 1 if index.numel() else 0
        scores = values.reshape(items, prod(values.shape[1:]))
        if exp_rescale is not None:
            shape = (1,) * (src.ndim - exp_rescale.ndim) + exp_rescale.shape
            weight = exp_rescale.reshape(shape).movedim(dim, 0)
            if prod(weight.shape[1:]) == 1:
                exp_rescale = weight.reshape(weight.shape[0], 1).expand(items, 1)
            else:
                exp_rescale = weight.expand_as(values).reshape_as(scores)
        out = graph_softmax(
            scores, index, num_nodes, exp_rescale, self.eps, backend=self.backend
        )
        return out.reshape_as(values).movedim(0, dim)
