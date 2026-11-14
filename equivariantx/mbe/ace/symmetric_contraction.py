"""Uuu symmetric contractions and fully connected coefficient maps."""

from collections import defaultdict

import torch
from e3nn import o3

from ...o3.linear import ElementLinear
from ...ops import resolve_backend
from ...ops.precision import validate_module_precision, validate_precision
from ...ops.torch.mbe.ace import symmetric_contraction as torch_kernel


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


class UuuSymmetricContraction(torch.nn.Module):
    """Contract repeated features with sparse symmetric coupling tensors.

    Parameters
    ----------
    irreps_in, irreps_out : o3.Irreps
        Input and output representations with equal multiplicities.
    coupling_tensors : sequence of tuple
        Entries ``(i_out, degree, tensor)``. A tensor has shape
        ``(irrep_out.dim, input_dim, ..., input_dim, num_paths)`` with
        ``degree`` input axes. Scalar output axes may be omitted.
        ``input_dim`` counts angular components without multiplicities.
        Weight blocks follow the supplied order.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        CPU tensors use PyTorch operations.
    layout : {"mul_ir", "ir_mul"}, optional
        Output layout. Defaults to ``"mul_ir"``.

    Examples
    --------
    >>> import torch
    >>> from equivariantx.mbe.ace import UuuSymmetricContraction
    >>> coupling = torch.ones(1, 1, 1, 1)
    >>> layer = UuuSymmetricContraction("2x0e", "2x0e", [(0, 2, coupling)])
    >>> x = torch.randn(3, 2, 1)
    >>> weight = torch.randn(2, 1, 2)
    >>> layer(x, weight, torch.tensor([0, 1, 0])).shape
    torch.Size([3, 2])
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        coupling_tensors,
        backend=None,
        layout="mul_ir",
    ):
        backend = resolve_backend(backend)
        super().__init__()
        irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        input_dim = sum(ir.dim for _, ir in irreps_in)
        offsets, offset = [], 0
        for _, ir in irreps_out:
            offsets.append(offset)
            offset += ir.dim
        entries, num_paths = [], 0
        for output, degree, tensor in coupling_tensors:
            if degree < 1 or not 0 <= output < len(irreps_out):
                raise ValueError("Provide a positive degree and a valid output index.")
            tensor = tensor.detach().to(device="cpu", dtype=torch.float64)
            paths = tensor.shape[-1]
            tensor = tensor.reshape(
                irreps_out[output].ir.dim, *([input_dim] * degree), paths
            )
            for index in tensor.nonzero().tolist():
                entries.append(
                    (
                        tuple(index[1:-1]),
                        num_paths + index[-1],
                        offsets[output] + index[0],
                        tensor[tuple(index)].item(),
                    )
                )
            num_paths += paths
        self.initialize(irreps_in, irreps_out, entries, num_paths, backend, layout)

    @classmethod
    def from_entries(
        cls, irreps_in, irreps_out, entries, num_paths, backend=None, layout="mul_ir"
    ):
        """Construct from sparse coupling coefficients.

        Parameters
        ----------
        entries : sequence of tuple
            ``(input_indices, weight_index, output_index, value)`` entries.
            Angular indices exclude channel multiplicities.
        num_paths : int
            Number of independent coefficient vectors.

        Returns
        -------
        UuuSymmetricContraction
            Module with the supplied representations, backend and layout.
        """
        backend = resolve_backend(backend)
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module.initialize(irreps_in, irreps_out, entries, num_paths, backend, layout)
        return module

    def initialize(self, irreps_in, irreps_out, entries, num_paths, backend, layout):
        """Register the symmetric polynomial and its reference tensors."""
        self.irreps_in, self.irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        multiplicities = {mul for mul, _ in (*self.irreps_in, *self.irreps_out)}
        if len(multiplicities) != 1 or min(multiplicities) < 1:
            raise ValueError("All multiplicities must agree and be positive.")
        if backend not in ("cuda", "torch") or layout not in ("mul_ir", "ir_mul"):
            raise ValueError("Invalid contraction backend or output layout.")
        self.backend, self.layout = backend, layout
        self.num_channels = multiplicities.pop()
        self.input_dim = sum(ir.dim for _, ir in self.irreps_in)
        self.output_dims = tuple(ir.dim for _, ir in self.irreps_out)
        self.output_dim = sum(self.output_dims)
        self.num_paths = num_paths
        self.weight_numel = num_paths * self.num_channels
        grouped = defaultdict(float)
        for indices, weight, output, value in entries:
            if not indices or not all(0 <= i < self.input_dim for i in indices):
                raise ValueError("Input angular indices are out of range.")
            if not 0 <= weight < num_paths or not 0 <= output < self.output_dim:
                raise ValueError("Weight or output indices are out of range.")
            grouped[tuple(sorted(indices)), weight, output] += value
        rows = tuple((*key, value) for key, value in sorted(grouped.items()) if value)
        self.correlation = max((len(row[0]) for row in rows), default=1)
        self.has_terms = bool(rows)
        self.metadata = repr(
            (self.input_dim, num_paths, self.output_dim, self.correlation, rows)
        )
        for degree in range(1, self.correlation + 1):
            terms = [row for row in rows if len(row[0]) == degree]
            self.register_buffer(
                f"indices_{degree}",
                torch.tensor([row[0] for row in terms], dtype=torch.long).reshape(
                    -1, degree
                ),
            )
            self.register_buffer(
                f"weights_{degree}",
                torch.tensor([row[1] for row in terms], dtype=torch.long),
            )
            self.register_buffer(
                f"outputs_{degree}",
                torch.tensor([row[2] for row in terms], dtype=torch.long),
            )
            self.register_buffer(
                f"coefficients_{degree}",
                torch.tensor([row[3] for row in terms], dtype=torch.float64),
            )

    def forward(self, features, weight, node_type):
        """Evaluate element-dependent symmetric contractions.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(nodes, channels, input_dim)``.
        weight : torch.Tensor
            Shape ``(elements, num_paths, channels)``.
        node_type : torch.Tensor
            Integer element indices of shape ``(nodes,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(nodes, irreps_out.dim)`` in the requested output layout.
        """
        validate_module_precision(self, (features, weight, node_type))
        validate_precision((features, weight))
        if features.ndim != 3 or features.shape[1:] != (
            self.num_channels,
            self.input_dim,
        ):
            raise ValueError("Features must have shape (nodes, channels, input_dim).")
        if weight.ndim != 3 or weight.shape[1:] != (self.num_paths, self.num_channels):
            raise ValueError("Weights must have shape (elements, num_paths, channels).")
        if weight.dtype != features.dtype or weight.device != features.device:
            raise ValueError("Features and weights must share a dtype and device.")
        if node_type.shape != features.shape[:1] or node_type.device != features.device:
            raise ValueError(
                "Provide one element index per node on the feature device."
            )
        if node_type.dtype not in (torch.int32, torch.int64):
            raise TypeError("Element indices must be integers.")
        if features.is_cuda and self.backend == "cuda":
            from ...ops.cuda.mbe.ace import symmetric_contraction as cuda_kernel

            return cuda_kernel.uuu(self, features, weight, node_type)
        return torch_kernel.uuu(self, features, weight, node_type)

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}, correlation={self.correlation}, {self.num_paths} paths"


class UuuFullyConnectedSymmetricContraction(torch.nn.Module):
    """Uuu symmetric contractions with fully connected coefficient maps.

    Parameters
    ----------
    tensor_products : sequence of o3.TensorProduct
        Unweighted ``uuu`` products for correlation orders two and higher.
        All supplied paths and their normalization are retained.
    linears : sequence of o3.Linear
        External-weight coefficient maps for orders one through the maximum
        correlation, including the linear contribution of the original input.
        All maps must have the same output layout. Biases are applied by the caller.
        Weights are indexed by element regardless of ``shared_weights``.
    backend : {None, "cuda", "torch"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        CUDA contracts sparse CG entries without expanded product tensors and
        fuses the final product with its coefficient map. CPU inputs use torch.

    Notes
    -----
    Inputs use flattened ``mul_ir`` storage. The module owns no learned weights.
    Linear maps mix channels of equivalent irreps at each correlation order.
    Tensor products remain channel-wise and retain the supplied coupling paths.
    Both backends support higher derivatives. CUDA weight gradients use
    atomic reductions and are not bitwise deterministic.

    Examples
    --------
    >>> import torch
    >>> from equivariantx.mbe import ace
    >>> from e3nn import o3
    >>> irreps = "2x0e + 2x1o"
    >>> tp = o3.ElementwiseTensorProduct(irreps, irreps)
    >>> linears = [o3.Linear(layout, "2x0e", internal_weights=False,
    ...                      shared_weights=False)
    ...            for layout in (irreps, tp.irreps_out)]
    >>> layer = ace.UuuFullyConnectedSymmetricContraction([tp], linears, backend="torch")
    >>> weights = [torch.randn(2, linear.weight_numel) for linear in linears]
    >>> features = torch.randn(3, layer.input_dim)
    >>> node_type = torch.tensor([0, 1, 0])
    >>> layer(features, weights, node_type).shape
    torch.Size([3, 2])
    """

    def __init__(self, tensor_products, linears, backend=None):
        backend = resolve_backend(backend)
        super().__init__()
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
        self.output_dim = self.contraction_dim = linears[0].irreps_out.dim
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
            for index, ins in enumerate(tp.instructions):
                i, j, k = ins.i_in1, ins.i_in2, ins.i_out
                mul, ir1 = tp.irreps_in1[i]
                mul2, ir2 = tp.irreps_in2[j]
                mul3, ir3 = tp.irreps_out[k]
                if mul != mul2 or mul != mul3:
                    raise ValueError("uuu paths require equal channel counts.")
                sx, sy, sz = (
                    tp.irreps_in1.slices()[i].start,
                    tp.irreps_in2.slices()[j].start,
                    tp.irreps_out.slices()[k].start,
                )
                for a, b, c, coefficient in self.coupling_entries(tp, index):
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
        self.coefficient_maps = torch.nn.ModuleList(
            ElementLinear(linear, backend="torch")
            if getattr(linear, "output_basis", "cartesian") == "cartesian"
            else None
            for linear in linears
        )

    @staticmethod
    def coupling_entries(tensor_product, index):
        """Return nonzero angular coefficients for a coupling instruction."""
        path = tensor_product.instructions[index]
        degrees = (
            tensor_product.irreps_in1[path.i_in1].ir.l,
            tensor_product.irreps_in2[path.i_in2].ir.l,
            tensor_product.irreps_out[path.i_out].ir.l,
        )
        cg = o3.wigner_3j(*degrees, dtype=torch.float64) * path.path_weight
        return [(a, b, c, float(cg[a, b, c])) for a, b, c in cg.nonzero().tolist()]

    def project_product(self, features, stage):
        """Return the irreducible intermediate features."""
        return features

    def project_output(self, features):
        """Return the irreducible coefficient output."""
        return features

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
        validate_module_precision(self, (features, weights, node_type))
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
        if features.is_cuda and self.backend == "cuda":
            from ...ops.cuda.mbe.ace import symmetric_contraction as cuda_kernel

            return cuda_kernel.uuu_fully_connected(self, features, weights, node_type)
        return torch_kernel.uuu_fully_connected(self, features, weights, node_type)

    def linear(self, features, weight, node_type, stage):
        """Apply element coefficients without gathering per-node matrices."""
        if features.is_cuda and self.backend == "cuda":
            from ...ops.cuda.mbe.ace import symmetric_contraction as cuda_kernel

            return cuda_kernel.linear(self, features, weight, node_type, stage)
        return torch_kernel.linear(self, features, weight, node_type, stage)
