################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''Copy from Nequip-v0.15.0, avoid some unnecessary synchronization operations in MLIPs'''
import torch
from lightning.pytorch.strategies import DDPStrategy


class SimpleDDPStrategy(DDPStrategy):
    """
    Copy from Nequip-v0.15.0, avoid some unnecessary synchronization operations in MLIPs
    Effectively Lightning's :class:`~lightning.pytorch.strategies.DDPStrategy`, 
    but doing manual gradient syncs instead of using PyTorch's :class:`~torch.nn.parallel.DistributedDataParallel` wrapper.

    Example use in the config file:

    .. code-block:: yaml

      trainer:
        _target_: lightning.Trainer
        # other trainer arguments
        strategy:
          _target_: tace.utils.strategy.SimpleDDPStrategy
    """

    def configure_ddp(self) -> None:
        pass

    def post_backward(self, closure_loss: torch.Tensor) -> None:
        """
        Manual syncing of gradients after the backwards pass.
        """
        # cat all gradients into a single tensor for efficiency
        grad_tensors = []
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_tensors.append(param.grad.data.view(-1))

        if grad_tensors:
            # cat and reduce
            flat_grads = torch.cat(grad_tensors)
            # NOTE: averaging (i.e. summing and dividing by number of ranks) is consistent with PyTorch Lightning's `DDPStrategy`
            # in the training loop, we account for this by multiplying the loss by the number of ranks before the backwards call
            if torch.distributed.get_backend() == "gloo":
                torch.distributed.all_reduce(
                    flat_grads, op=torch.distributed.ReduceOp.SUM
                )
                flat_grads /= torch.distributed.get_world_size()
            else:
                torch.distributed.all_reduce(
                    flat_grads, op=torch.distributed.ReduceOp.AVG
                )

            # copy reduced gradients back
            offset = 0
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    numel = param.grad.numel()
                    param.grad.data.copy_(
                        flat_grads[offset : offset + numel].view_as(param.grad.data)
                    )
                    offset += numel