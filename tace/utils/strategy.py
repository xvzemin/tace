"""Gradient synchronization for models with coordinate derivatives."""

import os

import torch
from lightning.pytorch.strategies import DDPStrategy


def is_rank_0():
    """Return whether the current process is rank zero or runs without DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    return all(
        int(os.environ.get(name, 0)) == 0
        for name in (
            "NODE_RANK",
            "GROUP_RANK",
            "LOCAL_RANK",
            "SLURM_PROCID",
            "JSM_NAMESPACE_RANK",
        )
    )


class SimpleDDPStrategy(DDPStrategy):
    """Average parameter gradients after backward without wrapping the model.

    Gradients are grouped by device and dtype for asynchronous reduction.
    Based on NequIP 0.15.0's manual synchronization strategy.

    Examples
    --------

    .. code-block:: yaml

      trainer:
        _target_: lightning.Trainer
        strategy:
          _target_: tace.utils.strategy.SimpleDDPStrategy
    """

    def configure_ddp(self) -> None:
        pass

    def post_backward(self, closure_loss: torch.Tensor) -> None:
        """Reduce coalesced gradients without wrapping the force forward pass."""
        groups = {}
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                groups.setdefault((grad.device, grad.dtype), []).append(grad)
        if not groups:
            return
        summed = torch.distributed.get_backend() == "gloo"
        pending = []
        for gradients in groups.values():
            flat = torch.cat([grad.reshape(-1) for grad in gradients])
            work = torch.distributed.all_reduce(
                flat,
                op=torch.distributed.ReduceOp.SUM
                if summed
                else torch.distributed.ReduceOp.AVG,
                async_op=True,
            )
            pending.append((work, flat, gradients))
        for work, flat, gradients in pending:
            work.wait()
            if summed:
                flat.div_(torch.distributed.get_world_size())
            views = flat.split([grad.numel() for grad in gradients])
            torch._foreach_copy_(
                gradients, [view.view_as(grad) for view, grad in zip(views, gradients)]
            )
