"""Atomic cluster expansion with external element-dependent coefficients."""

from collections import defaultdict

import torch
from e3nn import o3

from .contraction import contract


def make_metadata(rows, dims, weighted):
    schedules = []
    for role in range(len(dims)):
        grouped = defaultdict(list)
        for record, coefficient in rows:
            channels = (
                record[4] * record[5]
                if weighted and role == 2
                else record[5]
                if weighted and role == 3
                else record[4]
            )
            stride = (
                1
                if weighted and role == 2
                else record[8]
                if role == 3
                else record[role + 6]
            )
            grouped[record[role], channels, stride].append((record, coefficient))
        schedules.append(
            tuple((key, tuple(entries)) for key, entries in sorted(grouped.items()))
        )
    return repr((dims, weighted, tuple(schedules)))


class ACE(torch.nn.Module):
    """Evaluate successive channel-wise products and coefficient contractions.

    Parameters
    ----------
    tensor_products : sequence of o3.TensorProduct
        Unweighted ``uuu`` products for correlation orders two and higher.
        All supplied paths and their normalization are retained.
    linears : sequence of o3.Linear
        External-weight coefficient maps for orders one through the maximum
        correlation, including the linear contribution of the original input.
        All maps must have the same output layout. Biases are applied by the caller.
    backend : {"cuda", "torch"}, optional
        CUDA contracts sparse CG entries without expanded product tensors and
        fuses the final product with its coefficient map. CPU inputs use torch.

    Notes
    -----
    Inputs use flattened ``mul_ir`` order. The module owns no learned weights.
    Intermediate correlation orders remain available for reuse. The highest
    order path features and per-node coefficient matrices are not materialized
    by the CUDA contraction. Transposed contractions support repeated
    differentiation, including parameter gradients of force losses.
    Weight gradients use atomic reductions and need not be bitwise repeatable.
    """

    def __init__(self, tensor_products, linears, backend="cuda"):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be 'cuda' or 'torch'")
        if not tensor_products or len(tensor_products) + 1 != len(linears):
            raise ValueError("Provide an order-one map and one map per product.")
        if any(linear.irreps_out != linears[0].irreps_out for linear in linears):
            raise ValueError("Coefficient maps must have the same output layout.")
        if linears[0].irreps_in.simplify() != tensor_products[0].irreps_in1.simplify():
            raise ValueError("The order-one map must match the input layout.")
        self.backend = backend
        # References do not register duplicate parameters or checkpoint keys.
        self.tensor_products = tuple(tensor_products)
        self.linears = tuple(linears)
        self.input_dim = tensor_products[0].irreps_in1.dim
        self.output_dim = linears[0].irreps_out.dim
        self.product_dims = tuple(tp.irreps_out.dim for tp in tensor_products)
        self.weight_numels = tuple(linear.weight_numel for linear in linears)
        self.metadata = []
        self.linear_metadata = []
        for linear in linears:
            if linear.internal_weights:
                raise ValueError("ACE coefficient maps require external weights.")
            rows, offset = [], 0
            for ins in linear.instructions:
                if ins.i_in < 0:
                    raise ValueError("Pass coefficient biases separately.")
                cin, ir = linear.irreps_in[ins.i_in]
                cout = linear.irreps_out[ins.i_out].mul
                sx = linear.irreps_in.slices()[ins.i_in].start
                sz = linear.irreps_out.slices()[ins.i_out].start
                for m in range(ir.dim):
                    rows.append(
                        (
                            (sx + m, 0, offset, sz + m, cin, cout, ir.dim, 0, ir.dim),
                            ins.path_weight,
                        )
                    )
                offset += cin * cout
            dims = (linear.irreps_in.dim, 1, linear.weight_numel, linear.irreps_out.dim)
            self.linear_metadata.append(make_metadata(rows, dims, True))
        for stage, (tp, linear) in enumerate(zip(tensor_products, linears[1:])):
            if any(
                ins.connection_mode != "uuu" or ins.has_weight
                for ins in tp.instructions
            ):
                raise ValueError("ACE requires unweighted uuu tensor products.")
            if tp.irreps_out.simplify() != linear.irreps_in.simplify():
                raise ValueError("Coefficient inputs must match the product output.")
            weighted = stage == len(tensor_products) - 1
            dims = (
                (
                    tp.irreps_in1.dim,
                    tp.irreps_in2.dim,
                    linear.weight_numel,
                    linear.irreps_out.dim,
                )
                if weighted
                else (tp.irreps_in1.dim, tp.irreps_in2.dim, tp.irreps_out.dim)
            )
            rows = []
            linear_paths = []
            offset = 0
            for ins in linear.instructions:
                if ins.i_in < 0:
                    raise ValueError("Pass coefficient biases separately.")
                linear_paths.append((ins.i_in, ins.i_out, offset, ins.path_weight))
                offset += ins.path_shape[0] * ins.path_shape[1]
            for ins in tp.instructions:
                i, j, k = ins.i_in1, ins.i_in2, ins.i_out
                mul, ir1 = tp.irreps_in1[i]
                mul2, ir2 = tp.irreps_in2[j]
                mul3, ir3 = tp.irreps_out[k]
                if mul != mul2 or mul != mul3:
                    raise ValueError("uuu paths require equal channel counts.")
                cg = (
                    o3.wigner_3j(ir1.l, ir2.l, ir3.l, dtype=torch.float64)
                    * ins.path_weight
                )
                sx, sy, sz = (
                    tp.irreps_in1.slices()[i].start,
                    tp.irreps_in2.slices()[j].start,
                    tp.irreps_out.slices()[k].start,
                )
                for a, b, c in cg.nonzero().tolist():
                    coefficient = float(cg[a, b, c])
                    if weighted:
                        for input_ir, out_ir, sw, scale in linear_paths:
                            linear_slice = linear.irreps_in.slices()[input_ir]
                            if linear_slice.start <= sz < linear_slice.stop:
                                cout = linear.irreps_out[out_ir].mul
                                sout = linear.irreps_out.slices()[out_ir].start
                                weight_offset = (
                                    sw + (sz - linear_slice.start) // ir3.dim * cout
                                )
                                rows.append(
                                    (
                                        (
                                            sx + a,
                                            sy + b,
                                            weight_offset,
                                            sout + c,
                                            mul,
                                            cout,
                                            ir1.dim,
                                            ir2.dim,
                                            ir3.dim,
                                        ),
                                        coefficient * scale,
                                    )
                                )
                    else:
                        rows.append(
                            (
                                (
                                    sx + a,
                                    sy + b,
                                    sz + c,
                                    0,
                                    mul,
                                    0,
                                    ir1.dim,
                                    ir2.dim,
                                    ir3.dim,
                                ),
                                coefficient,
                            )
                        )
            self.metadata.append(make_metadata(rows, dims, weighted))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # These references are deliberately not registered a second time.
        # Move their constants as well for standalone torch-backend use.
        for module in (*self.tensor_products, *self.linears):
            module._apply(fn)
        return self

    def forward(self, features, weights, node_type):
        """Return coefficient contributions from all correlation orders.

        Parameters
        ----------
        features : torch.Tensor
            Node features of shape ``(nodes, irreps_in.dim)``.
        weights : sequence of torch.Tensor
            Coefficients of shape ``(elements, weight_numel)`` for each order,
            including order one.
        node_type : torch.Tensor
            Element index for each node, or zeros for shared coefficients.
        """
        if len(weights) != len(self.weight_numels):
            raise ValueError("Provide weights for every correlation order.")
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Features must have shape (nodes, irreps_in.dim).")
        if (
            node_type.shape != features.shape[:1]
            or node_type.dtype != torch.int64
            or node_type.device != features.device
        ):
            raise ValueError("node_type must be an int64 vector on the feature device.")
        for numel, weight in zip(self.weight_numels, weights):
            if weight.ndim != 2 or weight.shape[1] != numel:
                raise ValueError("Weights must have shape (elements, weight_numel).")
            if weight.device != features.device or weight.dtype != features.dtype:
                raise ValueError("Weights and features must share a device and dtype.")
        corr = features
        output = self.linear(features, weights[0], node_type, 0)
        for stage, weight in enumerate(weights[1:]):
            if features.is_cuda and self.backend == "cuda":
                if stage == len(self.product_dims) - 1:
                    operands = [
                        corr,
                        features,
                        weight,
                        features.new_empty((0, self.output_dim)),
                    ]
                else:
                    operands = [
                        corr,
                        features,
                        features.new_empty((0, self.product_dims[stage])),
                    ]
                result = contract(
                    self.metadata[stage],
                    repr(((tuple(range(len(operands))), len(operands) - 1, 0),)),
                    node_type,
                    operands,
                )[0]
                if stage == len(self.product_dims) - 1:
                    output = output + result
                    continue
                corr = result
            else:
                corr = self.tensor_products[stage](corr, features)
            output = output + self.linear(corr, weight, node_type, stage + 1)
        return output

    def linear(self, features, weight, node_type, stage):
        """Apply element coefficients without gathering per-node matrices."""
        if not features.is_cuda or self.backend == "torch":
            return self.linears[stage](features, weight[node_type])
        operands = [
            features,
            features.new_ones((features.shape[0], 1)),
            weight,
            features.new_empty((0, self.output_dim)),
        ]
        return contract(
            self.linear_metadata[stage],
            repr((((0, 1, 2, 3), 3, 0),)),
            node_type,
            operands,
        )[0]
