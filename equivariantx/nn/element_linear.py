"""Indexed channel maps shared by the representation modules."""

import torch

from ..ops.torch.nn import element_linear as torch_kernel
from ..ops.precision import validate_module_precision


class ElementLinear(torch.nn.Module):
    """Evaluate declared Linear paths with indexed external coefficients."""

    def __init__(self, linear, backend, element_mode, layout):
        super().__init__()
        self.expand_weights = False
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be 'cuda' or 'torch'.")
        if element_mode not in ("single", "pair"):
            raise ValueError("element_mode must be 'single' or 'pair'.")
        self.backend = backend
        self.element_mode = element_mode
        self.layout = layout
        self.num_experts = 1
        self.irreps_in = linear.irreps_in
        self.irreps_out = linear.irreps_out
        self.input_dim = self.irreps_in.dim
        self.output_dim = self.irreps_out.dim
        self.weight_numel = linear.weight_numel
        slices_in, slices_out = self.irreps_in.slices(), self.irreps_out.slices()
        self.paths = []
        offset = 0
        for ins in linear.instructions:
            if ins.i_in < 0:
                raise ValueError("Pass an unbiased Linear; apply biases separately.")
            ci = self.irreps_in[ins.i_in].mul
            co = self.irreps_out[ins.i_out].mul
            dim = self.irreps_in[ins.i_in].ir.dim
            if tuple(ins.path_shape) != (ci, co):
                raise ValueError("ElementLinear requires channel-mixing Linear paths.")
            if ci and co:
                self.paths.append(
                    (
                        slices_in[ins.i_in].start,
                        offset,
                        slices_out[ins.i_out].start,
                        ci,
                        co,
                        dim,
                        float(ins.path_weight),
                    )
                )
            offset += ci * co
        self.set_metadata()

    def set_metadata(self):
        """Describe channel layout and independently weighted paths."""
        self.metadata = repr(
            (
                self.input_dim,
                self.output_dim,
                self.weight_numel,
                self.num_experts,
                tuple(self.paths),
                self.layout,
            )
        )

    def forward(self, features, weight, node_type, target_type=None):
        """Apply single- or two-endpoint element weights.

        Parameters
        ----------
        features : torch.Tensor
            Features of shape ``(samples, irreps_in.dim)``.
        weight : torch.Tensor
            Shape ``(elements, weight_numel)`` in single mode or
            ``(source_elements, target_elements, weight_numel)`` in pair
            mode. Expert maps insert a ``num_experts`` axis before the last
            axis. Each ordered element pair has independent weights.
        node_type : torch.Tensor
            Int64 element indices of shape ``(samples,)``. In pair mode,
            these are source indices.
        target_type : torch.Tensor, optional
            Int64 target element indices of shape ``(samples,)``. Required
            in pair mode and omitted in single mode.

        Returns
        -------
        torch.Tensor
            Features of shape ``(samples, irreps_out.dim)``.
        """
        validate_module_precision(self, (features, weight, node_type, target_type))
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Features must have shape (samples, irreps_in.dim).")
        pair = self.element_mode == "pair"
        if pair != (target_type is not None):
            raise ValueError("target_type must be supplied only in pair mode.")
        indices = (node_type, target_type) if pair else (node_type,)
        for value in indices:
            if value.shape != features.shape[:1] or value.dtype != torch.int64:
                raise ValueError(
                    "Element indices must be int64 vectors of length samples."
                )
            if value.device != features.device:
                raise ValueError(
                    "Element indices and features must share their device."
                )
        axes = 2 if pair else 1
        shape = (self.num_experts, self.weight_numel)
        if weight.ndim == axes + 1 and self.num_experts == 1:
            shape = (self.weight_numel,)
        if tuple(weight.shape[axes:]) != shape:
            raise ValueError("Weights must have element, expert and path dimensions.")
        if weight.device != features.device:
            raise ValueError("Features and weights must share their device.")
        fused = features.is_cuda and self.backend == "cuda"
        if weight.dtype != features.dtype:
            raise ValueError("Features and weights must share their dtype.")
        if fused:
            from ..ops.cuda.nn import element_linear as cuda_kernel

            return cuda_kernel.evaluate(self, features, weight, node_type, target_type)
        return torch_kernel.evaluate(self, features, weight, node_type, target_type)

    def extra_repr(self):
        return (
            f"{self.irreps_in} -> {self.irreps_out}, "
            f"element_mode={self.element_mode}, num_experts={self.num_experts}"
        )
