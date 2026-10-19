"""Atomic cluster expansion with external element-dependent coefficients."""

from collections import defaultdict

import torch
from e3nn import o3

from .. import co3
from ..co3.compact import Contraction, Linear
from ..co3.trace_free import trace_free_basis
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


class TACE(torch.nn.Module):
    """Evaluate successive channel-wise products and coefficient contractions.

    Parameters
    ----------
    tensor_products : sequence of o3.TensorProduct or co3.TensorProduct
        Unweighted ``uuu`` products for correlation orders two and higher.
        All supplied paths and their normalization are retained.
    linears : sequence of o3.Linear or co3.Linear
        External-weight coefficient maps for orders one through the maximum
        correlation, including the linear contribution of the original input.
        All maps must have the same output layout. Biases are applied by the caller.
    backend : {"cuda", "torch"}, optional
        CUDA contracts sparse coupling coefficients across all paths and
        fuses the final product with its coefficient map. CPU inputs use torch.

    Notes
    -----
    Inputs use flattened ``mul_ir`` storage. The module owns no learned weights.
    Both backends support higher derivatives. CUDA weight gradients use
    atomic reductions and are not bitwise deterministic.
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
        self.weight_numels = tuple(linear.weight_numel for linear in linears)
        self.metadata = []
        self.linear_metadata = []
        self.cartesian = isinstance(tensor_products[0], co3.TensorProduct)
        self.product_dims = tuple(
            tp.irreps_out.spherical().dim if self.cartesian else tp.irreps_out.dim
            for tp in tensor_products
        )
        self.contract_output_dim = (
            linears[0].irreps_out.spherical().dim if self.cartesian else self.output_dim
        )
        if self.cartesian:
            self.input_basis = co3.TraceFreeBasis(tensor_products[0].irreps_in1)
            self.output_basis = co3.TraceFreeBasis(linears[0].irreps_out)
            self.compact_products = torch.nn.ModuleList(
                Contraction(tp) for tp in tensor_products
            )
            self.compact_linears = torch.nn.ModuleList(
                Linear(linear) for linear in linears
            )
            groups, offset = [], 0
            for mul, l in self.output_basis.groups:
                matrix = trace_free_basis(l)[1]
                groups.append(
                    (
                        offset,
                        mul,
                        2 * l + 1,
                        3**l,
                        tuple(
                            (a, b, float(matrix[a, b]))
                            for a, b in matrix.nonzero().tolist()
                        ),
                    )
                )
                offset += mul * 3**l
            self.expansion_metadata = repr((self.output_dim, tuple(groups)))
        for linear in linears:
            if linear.internal_weights:
                raise ValueError("ACE coefficient maps require external weights.")
            lin_in = (
                linear.irreps_in.spherical() if self.cartesian else linear.irreps_in
            )
            lin_out = (
                linear.irreps_out.spherical() if self.cartesian else linear.irreps_out
            )
            rows, offset = [], 0
            for ins in linear.instructions:
                if ins.i_in < 0:
                    raise ValueError("Pass coefficient biases separately.")
                cin, ir = lin_in[ins.i_in]
                cout = lin_out[ins.i_out].mul
                sx = lin_in.slices()[ins.i_in].start
                sz = lin_out.slices()[ins.i_out].start
                for m in range(ir.dim):
                    rows.append(
                        (
                            (sx + m, 0, offset, sz + m, cin, cout, ir.dim, 0, ir.dim),
                            ins.path_weight,
                        )
                    )
                offset += cin * cout
            dims = (lin_in.dim, 1, linear.weight_numel, lin_out.dim)
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
            tp_in1, tp_in2, tp_out = (
                (ir.spherical() if self.cartesian else ir)
                for ir in (tp.irreps_in1, tp.irreps_in2, tp.irreps_out)
            )
            lin_in = (
                linear.irreps_in.spherical() if self.cartesian else linear.irreps_in
            )
            lin_out = (
                linear.irreps_out.spherical() if self.cartesian else linear.irreps_out
            )
            dims = (
                (
                    tp_in1.dim,
                    tp_in2.dim,
                    linear.weight_numel,
                    lin_out.dim,
                )
                if weighted
                else (tp_in1.dim, tp_in2.dim, tp_out.dim)
            )
            rows = []
            linear_paths = []
            offset = 0
            for ins in linear.instructions:
                if ins.i_in < 0:
                    raise ValueError("Pass coefficient biases separately.")
                linear_paths.append((ins.i_in, ins.i_out, offset, ins.path_weight))
                offset += ins.path_shape[0] * ins.path_shape[1]
            for path_index, ins in enumerate(tp.instructions):
                i, j, k = ins.i_in1, ins.i_in2, ins.i_out
                mul, ir1 = tp_in1[i]
                mul2, ir2 = tp_in2[j]
                mul3, ir3 = tp_out[k]
                if mul != mul2 or mul != mul3:
                    raise ValueError("uuu paths require equal channel counts.")
                if self.cartesian:
                    cg = (
                        Contraction.coefficients((ir1.l, ir2.l, ir3.l))
                        * tp.path_factors[path_index]
                    )
                else:
                    cg = (
                        o3.wigner_3j(ir1.l, ir2.l, ir3.l, dtype=torch.float64)
                        * ins.path_weight
                    )
                sx, sy, sz = (
                    tp_in1.slices()[i].start,
                    tp_in2.slices()[j].start,
                    tp_out.slices()[k].start,
                )
                for a, b, c in cg.nonzero().tolist():
                    coefficient = float(cg[a, b, c])
                    if weighted:
                        for input_ir, out_ir, sw, scale in linear_paths:
                            linear_slice = lin_in.slices()[input_ir]
                            if linear_slice.start <= sz < linear_slice.stop:
                                cout = lin_out[out_ir].mul
                                sout = lin_out.slices()[out_ir].start
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
                                    *((sx, sy, sz) if self.cartesian else ()),
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
        if self.cartesian:
            features = self.input_basis.pack(features)
        corr = features
        output = self.linear(features, weights[0], node_type, 0)
        for stage, weight in enumerate(weights[1:]):
            if features.is_cuda and self.backend == "cuda":
                if stage == len(self.product_dims) - 1:
                    operands = [
                        corr,
                        features,
                        weight,
                        features.new_empty((0, self.contract_output_dim)),
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
                corr = (
                    self.compact_products[stage](corr, features)
                    if self.cartesian
                    else self.tensor_products[stage](corr, features)
                )
            output = output + self.linear(corr, weight, node_type, stage + 1)
        if self.cartesian:
            if (
                output.is_cuda
                and self.backend == "cuda"
                and not self.output_basis.identity
            ):
                from ..conv.co3.projection import project

                blocks = [
                    block.reshape(output.shape[0], mul, 2 * l + 1).transpose(1, 2)
                    for block, (mul, l) in zip(
                        output.split(self.output_basis.widths, -1),
                        self.output_basis.groups,
                    )
                ]
                return project(self.expansion_metadata, blocks, False)[0]
            return self.output_basis.unpack(output)
        return output

    def linear(self, features, weight, node_type, stage):
        """Apply element coefficients without gathering per-node matrices."""
        if not features.is_cuda or self.backend == "torch":
            if self.cartesian:
                return self.compact_linears[stage](features, weight[node_type])
            return self.linears[stage](features, weight[node_type])
        operands = [
            features,
            features.new_ones((features.shape[0], 1)),
            weight,
            features.new_empty((0, self.contract_output_dim)),
        ]
        return contract(
            self.linear_metadata[stage],
            repr((((0, 1, 2, 3), 3, 0),)),
            node_type,
            operands,
        )[0]
