"""Channelwise O(2) convolutions with optional frame-free evaluation."""

import math

import torch
from e3nn import o3

from ...co2 import SphericalCoupling
from ...o2 import Irrep, WignerD
from ..o2_o3.convolution import O2O3TensorProductConv


class UuO2TensorProductConv(O2O3TensorProductConv):
    """Apply an externally weighted UuLinear and aggregate its global output.

    Parameters
    ----------
    frame_in, frame_out : LocalFrame
        Input and output frames. Global entries must have multiplicity
        ``linear.num_channel``. Duplicate entries and both spatial and time
        parities are supported. The output frame determines inverse scaling
        when local orders are truncated.
    linear : UuLinear
        Channelwise map between the frames' local representations. Its
        instruction order, weight layout and path normalization are preserved.
    backend : {"cuda", "torch"}, optional
        Execution backend. CUDA fuses directional couplings and aggregation
        without storing edge messages. ``"torch"`` uses PyTorch operations
        and automatic differentiation for both aligned and transverse forms.

    """

    def __init__(self, frame_in, linear, frame_out, *, backend="cuda"):
        torch.nn.Module.__init__(self)
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        if (
            frame_in.irreps_out != linear.irreps_in
            or frame_out.irreps_out != linear.irreps_out
        ):
            raise ValueError("UuLinear representations must match the local frames.")
        self.backend = backend
        self.irreps_in = frame_in.irreps_in
        self.irreps_out = frame_out.irreps_in
        self.output_dim = self.irreps_out.dim
        self.instructions = linear.instructions
        self.weight_numel = linear.weight_numel
        self.num_harmonics = 1
        channels = linear.num_channel
        self.num_channel = channels
        if any(mul != channels for mul, _ in self.irreps_in + self.irreps_out):
            raise ValueError("Every global multiplicity must equal num_channel.")
        lmax = max(frame_in.lmax, frame_out.lmax)
        self.frame = WignerD(lmax, lmax, method="recursive")
        self.degree_offsets = tuple(
            sum((2 * k + 1) ** 2 for k in range(l)) for l in range(lmax + 1)
        )
        locations = []
        for frame in (frame_in, frame_out):
            entries = {ir: [] for ir, _ in frame.irreps_out}
            for (_, ir), ir_slice in zip(frame.irreps_in, frame.irreps_in.slices()):
                parity = ir.p * (-1) ** ir.l
                for m in range(min(ir.l, frame.mmax) + 1):
                    local_ir = Irrep(m, parity if m == 0 else 0, getattr(ir, "t", 1))
                    if m == 0:
                        rows = ((ir.l, 1),)
                    elif frame.basis_change and parity == -1:
                        rows = ((ir.l - m, -1), (ir.l + m, 1))
                    else:
                        rows = ((ir.l + m, 1), (ir.l - m, 1))
                    entries[local_ir].append((ir_slice.start, ir.l, rows))
            locations.append(entries)

        self.path_data, self.sparse_paths = [], []
        scalar_paths = []
        offset = 0
        for ins in self.instructions:
            ir = linear.irreps_in[ins.i_in].ir
            inputs, outputs = locations[0][ir], locations[1][ir]
            for i, (start, l_in, rows_in) in enumerate(inputs):
                for j, (end, l_out, rows_out) in enumerate(outputs):
                    dim, dim_out = 2 * l_in + 1, 2 * l_out + 1
                    scale = ins.path_weight * math.sqrt(
                        dim_out / (2 * min(l_out, frame_out.mmax) + 1)
                    )
                    cg = torch.zeros(dim, dim_out, dtype=torch.float64, device="cpu")
                    for (a, sign_in), (b, sign_out) in zip(rows_in, rows_out):
                        cg[a, b] = scale * sign_in * sign_out
                    index = len(self.path_data)
                    self.register_buffer(
                        f"cg_{index}",
                        cg.to(torch.get_default_dtype()),
                        persistent=False,
                    )
                    if l_in == l_out == 0:
                        scalar_paths.append(index)
                    self.path_data.append(
                        (
                            "uvu",
                            (
                                start,
                                end,
                                channels,
                                channels,
                                dim,
                                dim_out,
                                self.degree_offsets[l_in],
                                self.degree_offsets[l_out],
                                offset + (i * len(outputs) + j) * channels,
                                0,
                            ),
                        )
                    )
                    self.sparse_paths.append(
                        tuple((a, b, float(cg[a, b])) for a, b in cg.nonzero().tolist())
                    )
            offset += math.prod(ins.path_shape)
        self.has_unweighted = False
        metadata = (
            tuple(self.path_data),
            self.weight_numel,
            False,
            tuple(self.sparse_paths),
        )
        self.kernel_metadata = repr(metadata)
        # These weights are independent at each order, not CG slices of a
        # single spherical harmonic. Use the general angular-generator rule.
        self.direction_metadata = repr((*metadata, tuple(scalar_paths)))

        # Pole CG slices form an orthogonal basis for order-preserving maps.
        # Transform the radial projection, not the edge features or their weights.
        pairs = {}
        for index, (_, path) in enumerate(self.path_data):
            pairs.setdefault(path[:2], []).append(index)
        groups = {}
        for indices in pairs.values():
            path = self.path_data[indices[0]][1]
            l_in, l_out = (path[4] - 1) // 2, (path[5] - 1) // 2
            matrices = torch.zeros(
                len(indices), path[4], path[5], dtype=torch.float64, device="cpu"
            )
            for i, index in enumerate(indices):
                for a, b, value in self.sparse_paths[index]:
                    matrices[i, a, b] = value
            degrees, coefficients = [], []
            reconstruction = torch.zeros_like(matrices)
            for l in range(abs(l_in - l_out), l_in + l_out + 1):
                cg = o3.wigner_3j(l_in, l, l_out, dtype=torch.float64, device="cpu")
                pole = cg[:, l, :]
                values = (matrices * pole).sum((1, 2)) / pole.square().sum()
                if values.abs().max() > 1e-14:
                    degrees.append((l, cg))
                    coefficients.append(values)
                    reconstruction += values[:, None, None] * pole
            if not torch.allclose(matrices, reconstruction, atol=2e-12, rtol=2e-12):
                raise ValueError("The local map must commute with rotations about y.")
            transform = torch.stack(coefficients)
            groups.setdefault(transform.shape, []).append((indices, degrees, transform))

        self.transverse_couplings = torch.nn.ModuleDict()
        self.weight_transforms = []
        weight_paths = []
        transverse_paths = []
        offset = 0
        for index, group in enumerate(groups.values()):
            indices = [
                self.path_data[i][1][8] // channels
                for paths, _, _ in group
                for i in paths
            ]
            transform = torch.stack([values for _, _, values in group])
            self.register_buffer(
                f"weight_index_{index}",
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"weight_transform_{index}",
                transform.to(torch.get_default_dtype()),
                persistent=False,
            )
            self.weight_transforms.append((tuple(transform.shape), transform.tolist()))
            for paths, degrees, transform in group:
                path = self.path_data[paths[0]][1]
                l_in, l_out = (path[4] - 1) // 2, (path[5] - 1) // 2
                for row, (l, cg) in zip(transform, degrees):
                    weight_paths.extend(
                        (
                            (self.path_data[i][1][8] // channels, offset // channels),
                            float(value),
                        )
                        for i, value in zip(paths, row)
                        if value != 0
                    )
                    name = f"{l_in}_{l}_{l_out}"
                    if name not in self.transverse_couplings:
                        self.transverse_couplings[name] = SphericalCoupling(
                            l_in, l, l_out, "norm"
                        )
                    transverse_paths.append(
                        (
                            path[0],
                            0,
                            path[1],
                            channels,
                            1,
                            path[4],
                            2 * l + 1,
                            path[5],
                            offset,
                            1.0,
                            tuple(
                                (a, b, c, float(cg[a, b, c]))
                                for a, b, c in cg.nonzero().tolist()
                            ),
                        )
                    )
                    offset += channels
        self.transverse_paths = tuple(transverse_paths)
        self.transverse_layout = (("uvu", channels),) * len(transverse_paths)
        self.transverse_weight_numel = offset
        self.weight_metadata = repr(
            (
                channels,
                (self.weight_numel // channels, offset // channels),
                tuple(weight_paths),
            )
        )
        self.transverse_metadata = repr(
            (self.transverse_paths, offset, False, ("transverse", "norm"))
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for index, (_, values) in enumerate(self.weight_transforms):
            buffer = getattr(self, f"weight_transform_{index}")
            buffer.copy_(torch.tensor(values, dtype=buffer.dtype, device=buffer.device))
        for index, entries in enumerate(self.sparse_paths):
            buffer = getattr(self, f"cg_{index}")
            value = torch.zeros(buffer.shape, dtype=torch.float64, device="cpu")
            for a, b, coefficient in entries:
                value[a, b] = coefficient
            buffer.copy_(value.to(buffer))
        return self

    def transform_weights(self, weights):
        """Express local order weights in the directional coupling basis."""
        if self.backend == "cuda" and weights.is_cuda:
            from ...kernels.channel_product import local_product

            return local_product(self.weight_metadata, weights)
        values = weights.reshape(
            weights.shape[0], self.weight_numel // self.num_channel, self.num_channel
        )
        result = []
        for index, (shape, _) in enumerate(self.weight_transforms):
            count, outputs, inputs = shape
            selected = values.index_select(1, getattr(self, f"weight_index_{index}"))
            selected = selected.reshape(
                weights.shape[0], count, inputs, self.num_channel
            )
            result.append(
                torch.einsum(
                    "goi,bgic->bgoc",
                    getattr(self, f"weight_transform_{index}"),
                    selected,
                ).reshape(weights.shape[0], count * outputs * self.num_channel)
            )
        return torch.cat(result, dim=-1) if result else weights[..., :0]

    def forward_wigner(
        self,
        features,
        radial,
        projection,
        wigner,
        cutoff,
        edge_index,
        num_nodes,
        *,
        vectors=None,
        radial_network=None,
    ):
        """Evaluate the aligned convolution using packed Wigner-D matrices.

        With vectors, the PyTorch backend constructs differentiable matrices
        from them. CUDA uses the supplied matrices and generator derivatives.
        """
        if (
            vectors is None
            or self.backend == "torch"
            or not features.is_cuda
            or radial_network is not None
        ):
            if vectors is not None:
                wigner = self.frame.forward_packed(vectors, method="recursive")
            result = super().forward(
                features,
                radial,
                projection,
                wigner,
                cutoff,
                edge_index,
                num_nodes,
                radial_network=radial_network,
            )
            return result if vectors is None else result + vectors.sum() * 0
        from ..o2_o3.geometry import direction_contraction

        operands = [
            features,
            radial,
            projection,
            wigner.detach(),
            cutoff,
            features.new_empty(1).expand(num_nodes, self.output_dim),
        ]
        return direction_contraction(
            self.direction_metadata,
            repr(((0, (0, 1, 2, 3, 3, 4, 5), False, ((6, 0),)),)),
            vectors,
            edge_index[0],
            edge_index[1],
            operands,
            self.backend == "cuda",
        )[0]

    def forward_transverse(
        self,
        features,
        radial,
        projection,
        cutoff,
        edge_index,
        num_nodes,
        vectors,
        *,
        radial_network=None,
    ):
        """Evaluate local order weights without constructing alignment matrices."""
        if radial_network is not None and not projection.numel():
            raise ValueError(
                "A streamed radial network requires its final linear weight "
                "as projection."
            )
        if projection.numel():
            projection = self.transform_weights(projection)
        else:
            radial = self.transform_weights(radial)
            projection = projection.new_empty((0, self.transverse_weight_numel))
        return super().forward_transverse(
            features,
            radial,
            projection,
            cutoff,
            edge_index,
            num_nodes,
            vectors,
            radial_network=radial_network,
        )

    def forward(
        self,
        features,
        radial,
        projection,
        wigner,
        cutoff,
        edge_index,
        num_nodes,
        *,
        vectors=None,
        radial_network=None,
    ):
        """Evaluate the convolution and its differentiable radial projection.

        Parameters
        ----------
        features : torch.Tensor
            Node features, ``(nodes, irreps_in.dim)``, in flattened ir_mul order.
        radial : torch.Tensor or tuple
            Radial features, ``(edges, channels)`` or ``(1, channels)``.
            With an empty projection, supply UuLinear weights directly.
            With ``radial_network``, ``(tensor, kind)`` partitions may use
            ``kind="edge"``, ``"source"`` or ``"target"``.
        projection : torch.Tensor
            Final radial weight matrix, ``(channels, weight_numel)``. An empty
            matrix of shape ``(0, weight_numel)`` selects direct path weights.
        wigner : torch.Tensor or None
            Packed full degree matrices, ``(edges, sum((2*l+1)**2))``. A
            leading dimension of one shares matrices across edges. Unused
            when vectors are supplied; pass None to avoid constructing frames.
        cutoff : torch.Tensor
            Scalar edge factors, ``(edges, 1)`` or ``(1, 1)``.
        edge_index : torch.Tensor
            Source and target indices, ``(2, edges)``.
        num_nodes : int
            Number of output nodes.
        vectors : torch.Tensor, optional
            Frame directions, ``(edges, 3)`` or ``(1, 3)``. When supplied,
            evaluate the same map without selecting a transverse basis.
            Otherwise use the supplied matrices as the rotation reference.
        radial_network : torch.nn.Module, optional
            Sequential network preceding ``projection``. CUDA streams its
            layers with the convolution and recomputes activations in backward.
            Supply a nonempty final projection when using this network.

        Returns
        -------
        torch.Tensor
            Aggregated features, ``(num_nodes, irreps_out.dim)``, in ir_mul order.
        """
        if radial_network is not None and (
            self.backend != "cuda" or not features.is_cuda
        ):
            from ..network import materialize

            radial = radial_network(materialize(radial, edge_index))
            if radial.shape[-1] + 1 == projection.shape[0]:
                radial = torch.cat((radial, torch.ones_like(radial[:, :1])), -1)
            radial_network = None
        if vectors is not None:
            return self.forward_transverse(
                features,
                radial,
                projection,
                cutoff,
                edge_index,
                num_nodes,
                vectors,
                radial_network=radial_network,
            )
        return self.forward_wigner(
            features,
            radial,
            projection,
            wigner,
            cutoff,
            edge_index,
            num_nodes,
            vectors=vectors,
            radial_network=radial_network,
        )
