"""Equivariant linear maps with shared or element-dependent weights."""

import math

import torch
from e3nn import o3

from .contraction import contract


class Linear(torch.nn.Module):
    """Equivariant linear map with e3nn weights and normalization.

    Parameters
    ----------
    irreps_in, irreps_out : e3nn.o3.Irreps or str
        Input and output representations.
    internal_weights, shared_weights : bool, optional
        Store trainable weights and share them over leading dimensions.
    instructions : list of tuple of int, optional
        Allowed input/output irrep pairs.
    biases : bool or list of bool, optional
        Add biases to scalar outputs.
    path_normalization : {"element", "path"}, optional
        Normalization of coupling paths.
    backend : {"cuda", "torch"}, optional
        Indexed CUDA contractions or PyTorch operations. CPU uses PyTorch.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        internal_weights=None,
        shared_weights=None,
        instructions=None,
        biases=False,
        path_normalization="element",
        backend="cuda",
    ):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be 'cuda' or 'torch'.")
        self.backend = backend
        self.shared_weights = True if shared_weights is None else shared_weights
        self.internal_weights = (
            self.shared_weights if internal_weights is None else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("Internal weights must be shared.")
        description = o3.Linear(
            irreps_in,
            irreps_out,
            internal_weights=False,
            instructions=instructions,
            path_normalization=path_normalization,
        )
        self.irreps_in, self.irreps_out = description.irreps_in, description.irreps_out
        self.input_dim, self.output_dim = self.irreps_in.dim, self.irreps_out.dim
        self.weight_numel = description.weight_numel
        self.instructions = description.instructions
        self.reference = description
        self.contraction = ElementLinear(description)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in self.irreps_out]
        if len(biases) != len(self.irreps_out) or any(
            use and not ir.is_scalar() for use, (_, ir) in zip(biases, self.irreps_out)
        ):
            raise ValueError("Biases are only supported for invariant scalar outputs.")
        index, mask, offset = [0] * self.irreps_out.dim, [0.0] * self.irreps_out.dim, 0
        for use, section in zip(biases, self.irreps_out.slices()):
            if use:
                width = section.stop - section.start
                index[section] = range(offset, offset + width)
                mask[section] = [1.0] * width
                offset += width
        self.bias_numel = offset
        for name, count in (("weight", self.weight_numel), ("bias", self.bias_numel)):
            if self.internal_weights and count:
                self.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.randn(count) if name == "weight" else torch.zeros(count)
                    ),
                )
            else:
                self.register_buffer(name, torch.empty(0))
        self.register_buffer(
            "bias_index", torch.tensor(index, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "bias_scale", torch.tensor(mask, dtype=torch.float64), persistent=False
        )

    def forward(self, features, weight=None, bias=None):
        """Apply the map to features of shape ``(..., irreps_in.dim)``."""
        if weight is None:
            if self.weight_numel and not self.internal_weights:
                raise RuntimeError("External weights are required.")
            weight = self.weight
        if bias is None:
            if self.bias_numel and not self.internal_weights:
                raise RuntimeError("External biases are required.")
            bias = self.bias
        if features.ndim < 1 or features.shape[-1] != self.input_dim:
            raise ValueError("Features must end with irreps_in.dim.")
        shape = features.shape[:-1]
        rows = 1 if self.shared_weights else math.prod(shape)
        if (
            weight.numel() != rows * self.weight_numel
            or bias.numel() != rows * self.bias_numel
        ):
            raise ValueError("Weights and biases do not match the linear layout.")
        features = features.reshape(math.prod(shape), self.input_dim)
        types = (
            torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
            if self.shared_weights
            else torch.arange(features.shape[0], device=features.device)
        )
        rows = 1 if self.shared_weights else features.shape[0]
        if self.shared_weights and (not features.is_cuda or self.backend == "torch"):
            result = self.reference(features, weight)
        else:
            self.contraction.backend = self.backend
            result = self.contraction(
                features, weight.reshape(rows, self.weight_numel), types
            )
        if self.bias_numel:
            result = result + bias.reshape(rows, self.bias_numel).index_select(
                -1, self.bias_index
            ) * self.bias_scale.to(result)
        return result.reshape(shape + (self.output_dim,))

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights"


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
