"""Element-dependent linear maps with external weights."""

import torch

from .contraction import contract


class ElementLinear(torch.nn.Module):
    """Apply element-dependent weights without expanding them over nodes.

    Parameters
    ----------
    linear : o3.Linear
        Unbiased linear map defining the irreps, paths and normalization.
        Weights are supplied to forward, not copied from this map.
    backend : {"cuda", "torch"}, optional
        CUDA uses indexed contractions. CPU inputs use torch operations.

    Notes
    -----
    Features use flattened ``mul_ir`` layout. The module has no parameters
    or persistent buffers. Biases are applied separately by the caller.
    Transposed contractions support repeated differentiation. Weight
    gradients use atomic reductions and are not bitwise deterministic.
    """

    def __init__(self, linear, backend="cuda"):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be 'cuda' or 'torch'")
        self.backend = backend
        self.num_experts = 1
        self.irreps_in = linear.irreps_in
        self.irreps_out = linear.irreps_out
        self.input_dim = self.irreps_in.dim
        self.output_dim = self.irreps_out.dim
        self.weight_numel = linear.weight_numel
        self.paths = []
        offset = 0
        for ins in linear.instructions:
            if ins.i_in < 0:
                raise ValueError("Pass biases separately from the linear map.")
            ci, ir = self.irreps_in[ins.i_in]
            co = self.irreps_out[ins.i_out].mul
            if not ci or not co:
                continue
            self.paths.append(
                (
                    linear.irreps_in.slices()[ins.i_in].start,
                    offset,
                    linear.irreps_out.slices()[ins.i_out].start,
                    ci,
                    co,
                    ir.dim,
                    float(ins.path_weight),
                )
            )
            offset += ci * co
        self.metadata = repr(
            (
                self.irreps_in.dim,
                self.irreps_out.dim,
                self.weight_numel,
                self.num_experts,
                tuple(self.paths),
            )
        )

    def forward(self, features, weight, node_type):
        """Apply the linear map.

        Parameters
        ----------
        features : torch.Tensor
            Node features of shape ``(nodes, irreps_in.dim)``.
        weight : torch.Tensor
            Element weights of shape ``(elements, weight_numel)``. For MoE
            maps, the shape is ``(elements, num_experts, weight_numel)``.
        node_type : torch.Tensor
            Element indices of shape ``(nodes,)``, with dtype int64.

        Returns
        -------
        torch.Tensor
            Features of shape ``(nodes, irreps_out.dim)``.
        """
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Features must have shape (nodes, irreps_in.dim).")
        if node_type.shape != features.shape[:1] or node_type.dtype != torch.int64:
            raise ValueError("node_type must be an int64 vector of length nodes.")
        expected = (
            (self.weight_numel,)
            if weight.ndim == 2 and self.num_experts == 1
            else (self.num_experts, self.weight_numel)
        )
        if tuple(weight.shape[1:]) != expected:
            raise ValueError("Weights must have element, expert and path dimensions.")
        if (
            weight.dtype != features.dtype
            or weight.device != features.device
            or node_type.device != features.device
        ):
            raise ValueError(
                "Features and weights must share a dtype and device; indices must share their device."
            )
        weight = weight.reshape(weight.shape[0], self.num_experts, self.weight_numel)
        if features.is_cuda and self.backend == "cuda":
            return contract(
                self.metadata,
                2,
                node_type,
                features,
                weight,
                features.new_empty((0, self.output_dim)),
            )
        # The zero dependency also defines derivatives for disconnected maps.
        result = features.new_zeros((features.shape[0], self.output_dim))
        result = result + (features.sum() + weight.sum()) * 0
        selected = weight[node_type]
        for sx, sw, sy, ci, co, dim, scale in self.paths:
            x = features[:, sx : sx + self.num_experts * ci * dim].reshape(
                features.shape[0], self.num_experts, ci, dim
            )
            w = selected[:, :, sw : sw + ci * co].reshape(
                features.shape[0], self.num_experts, ci, co
            )
            value = scale * torch.einsum("neum,neuv->nevm", x, w)
            result[:, sy : sy + self.num_experts * co * dim] = result[
                :, sy : sy + self.num_experts * co * dim
            ] + value.flatten(1)
        return result

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}, num_experts={self.num_experts}"


class MoEElementLinear(ElementLinear):
    """Apply independent element-dependent linear maps to expert channels.

    Parameters
    ----------
    linear : o3.Linear
        Unbiased map for a single expert, including path normalization.
    num_experts : int
        Number of independent experts. Within each irrep, expert channels
        are contiguous and precede the angular dimension.
    backend : {"cuda", "torch"}, optional
        Execution backend. CPU inputs use torch operations.

    Notes
    -----
    Expert outputs remain separate. Routing or mixing experts is the
    caller's responsibility. See ElementLinear for forward arguments.
    """

    def __init__(self, linear, num_experts, backend="cuda"):
        super().__init__(linear, backend)
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        self.input_dim *= num_experts
        self.output_dim *= num_experts
        self.irreps_in = type(linear.irreps_in)(
            [(mul * num_experts, ir) for mul, ir in linear.irreps_in]
        )
        self.irreps_out = type(linear.irreps_out)(
            [(mul * num_experts, ir) for mul, ir in linear.irreps_out]
        )
        self.paths = [
            (sx * num_experts, sw, sy * num_experts, ci, co, dim, scale)
            for sx, sw, sy, ci, co, dim, scale in self.paths
        ]
        self.metadata = repr(
            (
                self.irreps_in.dim,
                self.irreps_out.dim,
                self.weight_numel,
                num_experts,
                tuple(self.paths),
            )
        )
