"""A contiguous gradient workspace for the complete joint training program."""

import torch
import torch.distributed as dist

from equivariantx.utils.autograd import clip_gradient_norm


class GradientWorkspace:
    """Attach explicit joint gradients, synchronize and clip in flat storage.

    Parameters
    ----------
    parameters : iterable of torch.nn.Parameter
        Trainable parameters in the joint program's output order. All workers
        must use the same parameters and unused-gradient mask.

    Notes
    -----
    Each attachment replaces the previous gradients. Gradient accumulation,
    AMP scaling and optimizer updates remain caller responsibilities. Copying
    into disjoint views preserves independence when compiled outputs alias.
    The optimizer still sees each original complete matrix.
    """

    def __init__(self, parameters):
        self.parameters = tuple(parameters)
        if not self.parameters or any(not p.requires_grad for p in self.parameters):
            raise ValueError("Supply nonempty trainable parameters in output order")
        groups = {}
        for index, parameter in enumerate(self.parameters):
            groups.setdefault((parameter.device, parameter.dtype), []).append(index)
        self.buffers, self.groups = [], []
        self.views = [None] * len(self.parameters)
        for (device, dtype), indices in groups.items():
            flat = torch.zeros(
                sum(self.parameters[i].numel() for i in indices),
                device=device,
                dtype=dtype,
            )
            offset = 0
            for index in indices:
                parameter = self.parameters[index]
                self.views[index] = flat[offset : offset + parameter.numel()].view_as(
                    parameter
                )
                offset += parameter.numel()
            self.buffers.append(flat)
            self.groups.append(indices)

    @torch.no_grad()
    def attach(self, gradients):
        """Copy one full set of explicit gradients into disjoint live views."""
        if len(gradients) != len(self.parameters):
            raise ValueError("Joint gradients do not match the parameter contract")
        for indices in self.groups:
            destinations, values = [], []
            for index in indices:
                gradient = gradients[index]
                if gradient is None:
                    self.views[index].zero_()
                    self.parameters[index].grad = None
                    continue
                if gradient.shape != self.parameters[index].shape:
                    raise ValueError(f"Gradient shape mismatch at parameter {index}")
                destinations.append(self.views[index])
                values.append(gradient)
                self.parameters[index].grad = self.views[index]
            if values:
                torch._foreach_copy_(destinations, values)

    def attach_loss(self, loss):
        """Consume a scalar TACE joint loss with precomputed parameter gradients.

        Only the unscaled ``ParameterLoss`` contract is accepted. This bypasses
        the outer gradient-attachment backward; force and stress derivatives
        have already been evaluated inside the compiled joint program.
        """
        backward = loss.grad_fn
        if type(backward).__name__ != "ParameterLossBackward":
            raise ValueError("Expected an unscaled explicit joint ParameterLoss")
        owners = tuple(
            function.variable
            for function, _ in backward.next_functions
            if function is not None and hasattr(function, "variable")
        )
        if len(owners) != len(self.parameters) or any(
            a is not b for a, b in zip(owners, self.parameters)
        ):
            raise ValueError("Joint loss parameter order differs from the workspace")
        self.attach(backward.saved_tensors)

    def detach(self):
        """Remove parameter gradients still owned by this workspace."""
        for parameter, view in zip(self.parameters, self.views):
            if parameter.grad is view:
                parameter.grad = None

    @torch.no_grad()
    def synchronize(self):
        """Average the complete gradient using the current distributed group."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        summed = dist.get_backend() == "gloo"
        op = dist.ReduceOp.SUM if summed else dist.ReduceOp.AVG
        requests = [
            dist.all_reduce(value, op=op, async_op=True) for value in self.buffers
        ]
        for request in requests:
            request.wait()
        if summed:
            torch._foreach_mul_(self.buffers, 1.0 / dist.get_world_size())

    @torch.no_grad()
    def clip(self, maximum):
        """Clip active gradients and return their original global L2 norm."""
        return clip_gradient_norm(self.parameters, maximum)
