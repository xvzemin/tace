"""Shared layouts and weights for channel-wise spherical tensor products."""

import math

import torch
from e3nn import o3

from ..ops import resolve_backend
from ..ops.common.sparse_coefficients import bilinear_tables
from ..ops.precision import validate_module_precision
from ..ops.torch.o3 import tensor_product as torch_kernel

__all__ = ["FactorizedTensorProduct"]

SPARSE_PRODUCT_SIZE = 32768
LITERAL_PRODUCT_SIZE = 64


def irrep_layout(irreps):
    """Group equal irreps without mixing their multiplicity indices."""
    indices = {}
    for (mul, ir), section in zip(irreps, irreps.slices()):
        if mul:
            indices.setdefault(ir, []).extend(range(section.start, section.stop))
    multiplicities = {len(value) // ir.dim for ir, value in indices.items()}
    if len(multiplicities) != 1:
        raise ValueError(
            "Each input and output must have uniform, positive multiplicity."
        )
    mul = multiplicities.pop()
    return mul, {
        ir: torch.tensor(indices[ir], dtype=torch.long, device="cpu").reshape(
            mul, ir.dim
        )
        for ir in sorted(indices)
    }


class FactorizedTensorProduct(torch.nn.Module):
    """Shared layout and weight handling for channel-wise products."""

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        irrep_normalization,
        internal_weights,
        shared_weights,
        backend=None,
    ):
        backend = resolve_backend(backend)
        super().__init__()
        self.backend = backend
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.input_dim1 = self.irreps_in1.dim
        self.input_dim2 = self.irreps_in2.dim
        self.output_dim = self.irreps_out.dim
        if irrep_normalization not in ("component", "norm", "none"):
            raise ValueError("irrep_normalization must be component, norm, or none.")
        self.irrep_normalization = irrep_normalization
        self.shared_weights = True if shared_weights is None else shared_weights
        self.internal_weights = (
            self.shared_weights if internal_weights is None else internal_weights
        )
        if self.internal_weights and not self.shared_weights:
            raise ValueError("Internal weights must be shared.")
        self.layouts = tuple(
            irrep_layout(irreps)
            for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        )
        mul1, mul2, mul_out = (item[0] for item in self.layouts)
        self.num_channels = mul_out
        if mul_out != max(mul1, mul2) or any(
            mul not in (1, mul_out) for mul in (mul1, mul2)
        ):
            raise ValueError("Input multiplicities must match the output or equal one.")
        self.constants = {}

    def add_constant(self, name, value):
        """Register a constant while retaining its construction precision."""
        self.constants[name] = value
        self.register_buffer(
            name,
            value.to(
                dtype=torch.get_default_dtype()
                if value.is_floating_point()
                else value.dtype,
                device=torch.get_default_device(),
                copy=True,
            ),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, value in self.constants.items():
            self._buffers[name] = value.to(self._buffers[name], copy=True)
        return self

    def build(self):
        """Prepare sector transforms, factor indices and output normalization."""
        self.inputs = []
        offset = 0
        for number, (mul, layout) in enumerate(self.layouts[:2]):
            weights = {
                ir: torch.arange(offset + i * mul, offset + (i + 1) * mul, device="cpu")
                for i, ir in enumerate(layout)
            }
            offset += len(layout) * mul
            sectors = {}
            for ir in layout:
                sectors.setdefault((self.sector(ir), getattr(ir, "t", 1)), []).append(
                    ir
                )
            inputs = []
            for sector, irreps in sorted(sectors.items()):
                name = f"input{number + 1}_{len(inputs)}"
                index = torch.cat([layout[ir] for ir in irreps], dim=-1)
                weight_index = torch.cat(
                    [weights[ir][:, None].expand(-1, ir.dim) for ir in irreps], dim=-1
                )
                scale = torch.cat(
                    [
                        torch.full(
                            (ir.dim,),
                            math.sqrt(ir.dim)
                            if self.irrep_normalization == "norm"
                            else 1.0,
                            dtype=torch.float64,
                            device="cpu",
                        )
                        for ir in irreps
                    ]
                )
                self.add_constant(f"{name}_index", index.flatten())
                self.add_constant(f"{name}_weight_index", weight_index.flatten())
                self.add_constant(f"{name}_scale", scale)
                self.add_constant(f"{name}_basis", self.input_basis(irreps))
                inputs.append((name, sector, tuple(irreps), mul, index.shape[-1]))
            self.inputs.append(tuple(inputs))

        mul, layout = self.layouts[2]
        routes = []
        squared_norms = {ir: 0.0 for ir in layout}
        for i, (_, sector1, irreps1, _, _) in enumerate(self.inputs[0]):
            for j, (_, sector2, irreps2, _, _) in enumerate(self.inputs[1]):
                outputs = []
                for ir in layout:
                    if (self.sector(ir), getattr(ir, "t", 1)) != (
                        sector1[0] * sector2[0],
                        sector1[1] * sector2[1],
                    ):
                        continue
                    squared_norm = sum(
                        self.reduced_coefficient(ir1.l, ir2.l, ir.l) ** 2
                        for ir1 in irreps1
                        for ir2 in irreps2
                        if ir.p == ir1.p * ir2.p
                        and getattr(ir, "t", 1)
                        == getattr(ir1, "t", 1) * getattr(ir2, "t", 1)
                        and abs(ir1.l - ir2.l) <= ir.l <= ir1.l + ir2.l
                    )
                    if squared_norm > 0:
                        squared_norms[ir] += squared_norm
                        outputs.append(ir)
                if outputs:
                    routes.append((i, j, tuple(outputs)))

        self.routes = []
        self.compact_routes = {}
        self.program_entries = {}
        mask = torch.zeros(self.irreps_out.dim, dtype=torch.float64, device="cpu")
        for i, j, irreps in routes:
            name = f"output_{len(self.routes)}"
            index = torch.cat([layout[ir] for ir in irreps], dim=-1)
            weight_index = torch.cat(
                [
                    torch.arange(
                        offset + k * mul, offset + (k + 1) * mul, device="cpu"
                    )[:, None].expand(-1, ir.dim)
                    for k, ir in enumerate(irreps)
                ],
                dim=-1,
            )
            offset += len(irreps) * mul
            scales = []
            for ir in irreps:
                scale = 1.0
                if self.irrep_normalization != "none":
                    scale = math.sqrt(
                        (ir.dim if self.irrep_normalization == "component" else 1.0)
                        / squared_norms[ir]
                    )
                scales.extend([scale] * ir.dim)
            mask[index.flatten()] = 1
            self.add_constant(f"{name}_index", index.flatten())
            self.add_constant(f"{name}_weight_index", weight_index.flatten())
            self.add_constant(
                f"{name}_scale", torch.tensor(scales, dtype=torch.float64, device="cpu")
            )
            self.add_constant(f"{name}_basis", self.output_basis(irreps))
            if getattr(self, "method", None) == "coefficients":
                self.add_constant(
                    f"{name}_coupling",
                    self.coupling_tensor(
                        self.inputs[0][i][2], self.inputs[1][j][2], irreps
                    ),
                )
            dimensions = (self.inputs[0][i][-1], self.inputs[1][j][-1], index.shape[-1])
            if math.prod(dimensions) <= SPARSE_PRODUCT_SIZE:
                entries = self.coupling_entries(
                    self.inputs[0][i][2], self.inputs[1][j][2], irreps
                )
                scales1 = self.constants[f"{self.inputs[0][i][0]}_scale"]
                scales2 = self.constants[f"{self.inputs[1][j][0]}_scale"]
                entries = tuple(
                    (a, b, c, coefficient * float(scales1[a] * scales2[b]) * scales[c])
                    for a, b, c, coefficient in entries
                )
                for k, table in enumerate(bilinear_tables(dimensions, entries)):
                    self.add_constant(f"{name}_sparse_{k}", table)
                self.compact_routes[name] = (
                    dimensions,
                    entries if len(entries) <= LITERAL_PRODUCT_SIZE else (),
                )
                self.program_entries[name] = entries
            self.routes.append((i, j, name, index.shape[-1]))
        self.add_constant("output_mask", mask)
        self.inputs = tuple(
            tuple((name, mul, dim) for name, _, _, mul, dim in entries)
            for entries in self.inputs
        )
        del self.layouts
        self.weight_numel = offset
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer("weight", torch.empty(0), persistent=False)

    def coupling_entries(self, irreps1, irreps2, irreps_out):
        """Exact nonzero spherical coefficients, before normalization."""
        entries = []
        a = 0
        for ir1 in irreps1:
            b = 0
            for ir2 in irreps2:
                c = 0
                for ir in irreps_out:
                    if (
                        ir.p == ir1.p * ir2.p
                        and getattr(ir, "t", 1)
                        == getattr(ir1, "t", 1) * getattr(ir2, "t", 1)
                        and abs(ir1.l - ir2.l) <= ir.l <= ir1.l + ir2.l
                    ):
                        coefficient = self.reduced_coefficient(ir1.l, ir2.l, ir.l)
                        if coefficient:
                            block = coefficient * o3.wigner_3j(
                                ir1.l, ir2.l, ir.l, dtype=torch.float64, device="cpu"
                            )
                            entries.extend(
                                (a + x, b + y, c + z, float(block[x, y, z]))
                                for x, y, z in block.nonzero().tolist()
                            )
                    c += ir.dim
                b += ir2.dim
            a += ir1.dim
        return tuple(entries)

    def forward(self, x, y, weight=None):
        """Evaluate the tensor product.

        Parameters
        ----------
        x, y : torch.Tensor
            Features of shape ``(..., irreps_in1.dim)`` and
            ``(..., irreps_in2.dim)`` in ``mul_ir`` layout. Leading dimensions
            are broadcast.
        weight : torch.Tensor, optional
            Factor weights of shape ``(weight_numel,)`` when shared, otherwise
            ``(..., weight_numel)``. Defaults to the internal parameters.

        Returns
        -------
        torch.Tensor
            Features of shape ``(..., irreps_out.dim)`` in ``mul_ir`` layout.
        """
        validate_module_precision(self, (x, y, weight))
        if x.ndim < 1 or x.shape[-1] != self.input_dim1:
            raise ValueError("The first input must end with irreps_in1.dim.")
        if y.ndim < 1 or y.shape[-1] != self.input_dim2:
            raise ValueError("The second input must end with irreps_in2.dim.")
        if weight is None:
            if not self.internal_weights:
                raise ValueError("External weights are required.")
            weight = self.weight
        if weight.ndim < 1 or weight.shape[-1] != self.weight_numel:
            raise ValueError("Weights must end with weight_numel.")
        if self.shared_weights and weight.ndim != 1:
            raise ValueError("Shared weights must be one-dimensional.")
        if getattr(self, "backend", "torch") == "cuda" and x.is_cuda:
            from ..ops.cuda.o3 import tensor_product as cuda_kernel

            return cuda_kernel.evaluate(self, x, y, weight)
        return torch_kernel.evaluate(self, x, y, weight)

    def extra_repr(self):
        return (
            f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}, "
            f"{self.weight_numel} weights"
        )


class TensorProductConv(torch.nn.Module):
    """Gather source features, couple edge attributes and sum at destinations."""

    def __init__(self, tensor_product, *, backend=None, radial_mode="last"):
        super().__init__()
        backend = resolve_backend(backend)
        if radial_mode not in ("last", "full"):
            raise ValueError("radial_mode must be last or full.")
        self.backend, self.radial_mode = backend, radial_mode
        self.tensor_product = tensor_product
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.weight_numel = tensor_product.weight_numel
        self.input_dim = self.irreps_in1.dim
        self.edge_dim = self.irreps_in2.dim
        self.output_dim = self.irreps_out.dim
        self.program_metadata = torch_kernel.convolution_program(tensor_product)

    def forward(
        self,
        features,
        edge_attrs,
        edge_index,
        weight=None,
        num_nodes=None,
        *,
        radial=None,
        radial_network=None,
    ):
        """Evaluate messages and sum them at destination nodes.

        Parameters
        ----------
        features : torch.Tensor
            Source-node features of shape ``(nodes, irreps_in1.dim)``.
        edge_attrs : torch.Tensor
            Edge features of shape ``(edges, irreps_in2.dim)``.
        edge_index : torch.Tensor
            Int64 source and destination indices of shape ``(2, edges)``.
        weight : torch.Tensor, optional
            Shared factors of shape ``(weight_numel,)`` or edge factors of shape
            ``(edges, weight_numel)``, according to the tensor product.
        num_nodes : int, optional
            Number of destination nodes. Defaults to the source-node count.
        radial : torch.Tensor or tuple, optional
            Inputs to radial_network. A tuple of ``(tensor, storage)`` pairs
            may use ``"edge"``, ``"source"`` and ``"target"`` storage.
        radial_network : torch.nn.Module, optional
            Network producing edge factors from radial inputs. Requires an
            externally weighted tensor product with shared_weights=False.
            Cannot be combined with weight.

        Returns
        -------
        torch.Tensor
            Destination features of shape ``(num_nodes, irreps_out.dim)``.
        """
        validate_module_precision(
            self,
            (
                features,
                edge_attrs,
                edge_index,
                weight,
                num_nodes,
                radial,
                radial_network,
            ),
        )
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("features must have shape (nodes, irreps_in1.dim).")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError("edge_index must be an int64 tensor of shape (2, edges).")
        if edge_attrs.shape != (edge_index.shape[1], self.edge_dim):
            raise ValueError("edge_attrs must have shape (edges, irreps_in2.dim).")
        if radial_network is not None:
            if weight is not None or radial is None:
                raise ValueError(
                    "Provide radial inputs instead of weight when using radial_network."
                )
            if (
                self.tensor_product.shared_weights
                or self.tensor_product.internal_weights
            ):
                raise ValueError("radial_network requires external, unshared weights.")
        elif radial is not None:
            raise ValueError("radial inputs require radial_network.")
        if (
            weight is not None
            and weight.ndim > 1
            and weight.shape != (edge_index.shape[1], self.weight_numel)
        ):
            raise ValueError("Edge weights must have shape (edges, weight_numel).")
        num_nodes = features.shape[0] if num_nodes is None else num_nodes
        if (
            weight is None
            and radial_network is None
            and not self.tensor_product.internal_weights
        ):
            raise ValueError("External weights are required.")
        if weight is not None and (
            weight.ndim < 1
            or weight.shape[-1] != self.weight_numel
            or self.tensor_product.shared_weights
            and weight.ndim != 1
        ):
            raise ValueError("Weights must match weight_numel and shared_weights.")
        kernel = torch_kernel
        if self.backend == "cuda" and features.is_cuda:
            from ..ops.cuda.o3 import tensor_product as kernel

        return kernel.convolve(
            self,
            features,
            edge_attrs,
            edge_index,
            weight,
            num_nodes,
            radial,
            radial_network,
        )

    def extra_repr(self):
        return f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}"
