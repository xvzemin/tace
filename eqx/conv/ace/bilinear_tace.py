"""Gated channel-wise products fused with element and expert coefficients."""

from collections import defaultdict

import torch
from e3nn import o3

from eqx import o3 as eqx_o3

from .bilinear_contraction import contract, encode_metadata


class BilinearTACE(torch.nn.Module):
    """Contract a gated bilinear product directly into its coefficient output.

    Parameters
    ----------
    tensor_product : o3.TensorProduct
        External-weight ``uuu`` tensor product. All paths are retained.
    linear : o3.Linear
        Unbiased external coefficient map, defined for one expert. Its input
        irreps may simplify adjacent product outputs.
    num_experts : int, optional
        Number of independent channel groups within each linear input irrep.
    backend : {"cuda", "torch"}, optional
        CUDA fuses the product, gate and coefficient contraction without
        storing product features or per-node coefficient matrices.
    shared_linear : o3.Linear, optional
        Element-independent map over all channels, evaluated from the same
        block-local angular product and added to the expert output.

    Notes
    -----
    Features use flattened ``mul_ir`` layout. Weights and biases remain owned
    by the caller. The order-one contribution is added by the caller.
    Expert and shared maps reuse a block-local angular tile. Their transposes
    are combined before differentiating the product inputs and gate, without
    a full product-gradient buffer. Recursive contractions support higher
    derivatives, including force training. Atomic reductions are used in
    backward and need not be bitwise deterministic.
    """

    def __init__(
        self, tensor_product, linear, num_experts=1, backend="cuda", shared_linear=None
    ):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be 'cuda' or 'torch'")
        if any(
            ins.connection_mode != "uuu" or not ins.has_weight
            for ins in tensor_product.instructions
        ):
            raise ValueError("BilinearTACE requires weighted uuu paths.")
        if tensor_product.internal_weights or linear.internal_weights:
            raise ValueError("Provide external product and coefficient weights.")
        full_input = o3.Irreps(
            [(mul * num_experts, ir) for mul, ir in linear.irreps_in]
        )
        if full_input.simplify() != tensor_product.irreps_out.simplify():
            raise ValueError("Coefficient inputs must match the product output.")
        self.backend = backend
        self.tensor_products = (tensor_product,)
        self.coefficients = eqx_o3.MoEElementLinear(
            linear, num_experts, backend=backend
        )
        self.input_dim = tensor_product.irreps_in1.dim
        self.base_dim = tensor_product.irreps_in2.dim
        self.output_dim = linear.irreps_out.dim * num_experts
        self.weight_numel = linear.weight_numel * num_experts
        self.gate_dim = tensor_product.weight_numel
        self.shared = (
            BilinearTACE(tensor_product, shared_linear, backend=backend)
            if shared_linear is not None
            else None
        )
        if self.shared is not None and self.shared.output_dim != self.output_dim:
            raise ValueError("Shared and expert coefficient outputs must match.")
        rows = []
        gate_offset = 0
        for ins in tensor_product.instructions:
            mul, ir1 = tensor_product.irreps_in1[ins.i_in1]
            mul2, ir2 = tensor_product.irreps_in2[ins.i_in2]
            mul3, ir3 = tensor_product.irreps_out[ins.i_out]
            if mul != mul2 or mul != mul3:
                raise ValueError("uuu paths require equal channel counts.")
            sx = tensor_product.irreps_in1.slices()[ins.i_in1].start
            sy = tensor_product.irreps_in2.slices()[ins.i_in2].start
            sz = tensor_product.irreps_out.slices()[ins.i_out].start
            cg = (
                o3.wigner_3j(ir1.l, ir2.l, ir3.l, dtype=torch.float64) * ins.path_weight
            )
            weight_offset = 0
            for path in linear.instructions:
                if path.i_in < 0:
                    raise ValueError("Pass biases separately.")
                ci, _ = linear.irreps_in[path.i_in]
                co = linear.irreps_out[path.i_out].mul
                sl = full_input.slices()[path.i_in]
                if sl.start < sz + mul * ir3.dim and sz < sl.stop:
                    start = (sz - sl.start) // ir3.dim
                    for expert in range(num_experts):
                        lower, upper = (
                            max(start, expert * ci),
                            min(start + mul, (expert + 1) * ci),
                        )
                        if lower >= upper:
                            continue
                        shift = lower - start
                        sw = (
                            expert * linear.weight_numel
                            + weight_offset
                            + (lower - expert * ci) * co
                        )
                        so = (
                            linear.irreps_out.slices()[path.i_out].start * num_experts
                            + expert * co * ir3.dim
                        )
                        for a, b, c in cg.nonzero().tolist():
                            record = (
                                sx + shift * ir1.dim + a,
                                sy + shift * ir2.dim + b,
                                sw,
                                so + c,
                                upper - lower,
                                co,
                                ir1.dim,
                                ir2.dim,
                                ir3.dim,
                                gate_offset + shift,
                                gate_offset,
                                float(path.path_weight),
                                float(cg[a, b, c]),
                                sx,
                                sy,
                                so,
                                a,
                                b,
                                c,
                            )
                            rows.append((record, float(cg[a, b, c]) * path.path_weight))
                weight_offset += ci * co
            gate_offset += mul
        schedules = []
        for role in range(5):
            grouped = defaultdict(list)
            for r, coefficient in rows:
                offset = r[9] if role == 4 else r[role]
                channels = r[4] * r[5] if role == 2 else r[5] if role == 3 else r[4]
                stride = 1 if role in (2, 4) else r[8] if role == 3 else r[role + 6]
                grouped[offset, channels, stride].append((r, coefficient))
            schedules.append(
                tuple((key, tuple(values)) for key, values in sorted(grouped.items()))
            )
        dims = (
            self.input_dim,
            self.base_dim,
            self.weight_numel,
            self.output_dim,
            self.gate_dim,
        )
        self.metadata = repr((dims, True, tuple(schedules)))
        self.kernel_metadata = encode_metadata(
            [self.metadata]
            if self.shared is None
            else [self.metadata, self.shared.metadata]
        )

    def forward(self, features, base, gates, weight, node_type, shared_weight=None):
        """Return bilinear coefficients from node features and external weights.

        Parameters
        ----------
        features, base : torch.Tensor
            Node arrays matching the two tensor-product input irreps.
        gates : torch.Tensor
            Per-node product weights of shape ``(nodes, gate_dim)``.
        weight : torch.Tensor
            Coefficients of shape ``(elements, experts, weight_numel)`` or
            ``(elements, weight_numel)`` for a single expert.
        node_type : torch.Tensor
            Element indices of shape ``(nodes,)``, with dtype int64.
        shared_weight : torch.Tensor, optional
            Coefficient vector for ``shared_linear``. Shared and expert
            contributions are summed without a normalization factor.
        """
        for value, dim in (
            (features, self.input_dim),
            (base, self.base_dim),
            (gates, self.gate_dim),
        ):
            if value.ndim != 2 or value.shape != (features.shape[0], dim):
                raise ValueError(
                    "Features, base and gates must match their input layouts."
                )
            if value.dtype != features.dtype or value.device != features.device:
                raise ValueError("Inputs must share a device and dtype.")
        if (
            node_type.dtype != torch.int64
            or node_type.shape != features.shape[:1]
            or node_type.device != features.device
        ):
            raise ValueError("node_type must be an int64 vector on the feature device.")
        if (
            weight.ndim not in (2, 3)
            or weight.numel() != weight.shape[0] * self.weight_numel
        ):
            raise ValueError("Coefficient weights must match the expert layout.")
        if weight.dtype != features.dtype or weight.device != features.device:
            raise ValueError("Weights and features must share a device and dtype.")
        if self.shared is not None:
            if (
                shared_weight is None
                or shared_weight.numel() != self.shared.weight_numel
            ):
                raise ValueError("Provide one shared coefficient vector.")
            if (
                shared_weight.dtype != features.dtype
                or shared_weight.device != features.device
            ):
                raise ValueError(
                    "Shared weights must match the feature device and dtype."
                )
        elif shared_weight is not None:
            raise ValueError("This module has no shared coefficient map.")
        if features.is_cuda and self.backend == "cuda":
            operands = [
                features,
                base,
                weight.flatten(1),
                features.new_empty((0, self.output_dim)),
                gates,
            ]
            program = [(0, (0, 1, 2, 3, 4), 3, 0)]
            if self.shared is not None:
                operands.append(shared_weight.reshape(1, -1))
                program.append((1, (0, 1, 5, 3, 4), 3, 0))
            return contract(
                self.kernel_metadata, repr(tuple(program)), node_type, operands
            )[0]
        product = self.tensor_products[0](features, base, gates)
        output = self.coefficients(product, weight, node_type)
        if self.shared is not None:
            output = output + self.shared.coefficients(
                product, shared_weight.reshape(1, 1, -1), torch.zeros_like(node_type)
            )
        return output

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.tensor_products[0]._apply(fn)
        return self
