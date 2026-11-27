################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Gradient synchronization for models with coordinate derivatives."""

import os
from contextlib import contextmanager

import torch
from lightning.pytorch.overrides.distributed import _sync_module_states
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

    Rank zero supplies the initial parameters and buffers. Dense gradients
    are averaged by device and dtype at the end of each accumulation window.
    Forward passes do not broadcast buffers.
    Based on NequIP 0.15.0's manual synchronization strategy.

    Examples
    --------

    .. code-block:: yaml

      trainer:
        _target_: lightning.Trainer
        strategy:
          _target_: tace.utils.strategy.SimpleDDPStrategy
    """

    _sync_gradients = True

    def configure_ddp(self) -> None:
        if torch.distributed.get_world_size() > 1:
            _sync_module_states(self.model)

    @contextmanager
    def block_backward_sync(self):
        previous = self._sync_gradients
        self._sync_gradients = False
        try:
            yield
        finally:
            self._sync_gradients = previous

    def post_backward(self, closure_loss: torch.Tensor) -> None:
        """Average gradients across replicas, preserving globally unused parameters."""
        if not self._sync_gradients or torch.distributed.get_world_size() == 1:
            return
        groups = {}
        for param in self.model.parameters():
            if param.requires_grad:
                groups.setdefault((param.device, param.dtype), []).append(param)
        summed = torch.distributed.get_backend() == "gloo"
        pending = []
        for params in groups.values():
            present = [p.grad is not None for p in params]
            if all(present):
                chunks = [p.grad.detach().reshape(-1) for p in params]
                flags = params[0].new_ones(1).expand(len(params))
            else:
                zero = params[0].new_zeros(1)
                chunks = [
                    p.grad.detach().reshape(-1) if used else zero.expand(p.numel())
                    for p, used in zip(params, present)
                ]
                flags = params[0].new_tensor(present)
            flat = torch.cat([*chunks, flags])
            work = torch.distributed.all_reduce(
                flat,
                op=torch.distributed.ReduceOp.SUM
                if summed
                else torch.distributed.ReduceOp.AVG,
                async_op=True,
            )
            pending.append((work, flat, params, present))
        for work, flat, params, present in pending:
            work.wait()
            if summed:
                flat.div_(torch.distributed.get_world_size())
            *values, flags = flat.split([*(p.numel() for p in params), len(params)])
            # No host read on the dense path. A missing local gradient needs
            # storage only when another replica contributed to this parameter.
            if not all(present):
                for p, used, used_anywhere in zip(params, present, flags.tolist()):
                    if not used and used_anywhere:
                        p.grad = torch.empty_like(p)
            out, src = [], []
            for p, value in zip(params, values):
                if p.grad is not None:
                    out.append(p.grad)
                    src.append(value.view_as(p))
            if out:
                torch._foreach_copy_(out, src)
