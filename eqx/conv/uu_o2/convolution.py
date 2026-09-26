"""Channelwise O(2) convolutions in aligned frames."""

import math

import torch

from ...o2 import Irrep
from ..o2_o3.convolution import O2O3TensorProductConv


class UuO2TensorProductConv(O2O3TensorProductConv):
    """Fuse frame rotations, an externally weighted UuLinear and aggregation.

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
        Execution backend. CUDA fuses gather, rotations, all local paths and
        scatter without storing edge messages. CPU uses tensor operations.

    Notes
    -----
    Each local path has one or two fixed angular coefficients. The frame
    basis changes are incorporated into these coefficients before compilation.
    The aligned contraction engine is shared with O2O3TensorProductConv; no
    Clebsch--Gordan expansion or restriction of the learned weights is needed.
    Radial projections use bounded, reusable workspaces that are not retained
    for backward. Transposed contractions support recursive higher derivatives.
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
        if any(mul != channels for mul, _ in self.irreps_in + self.irreps_out):
            raise ValueError("Every global multiplicity must equal num_channel.")
        lmax = max(frame_in.lmax, frame_out.lmax)
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
                    cg = torch.zeros(dim, dim_out)
                    for (a, sign_in), (b, sign_out) in zip(rows_in, rows_out):
                        cg[a, b] = scale * sign_in * sign_out
                    index = len(self.path_data)
                    self.register_buffer(f"cg_{index}", cg, persistent=False)
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
    ):
        """Evaluate the convolution and its differentiable radial projection.

        Parameters
        ----------
        features : torch.Tensor
            Node features, ``(nodes, irreps_in.dim)``, in flattened ir_mul order.
        radial : torch.Tensor
            Radial features, ``(edges, channels)`` or ``(1, channels)``.
            With an empty projection, supply UuLinear weights directly.
        projection : torch.Tensor
            Final radial weight matrix, ``(channels, weight_numel)``. An empty
            matrix of shape ``(0, weight_numel)`` selects direct path weights.
        wigner : torch.Tensor
            Packed full degree matrices, ``(edges, sum((2*l+1)**2))``. A
            leading dimension of one shares matrices across edges.
        cutoff : torch.Tensor
            Scalar edge factors, ``(edges, 1)`` or ``(1, 1)``.
        edge_index : torch.Tensor
            Source and target indices, ``(2, edges)``.
        num_nodes : int
            Number of output nodes.
        vectors : torch.Tensor, optional
            Frame directions, ``(edges, 3)`` or ``(1, 3)``. When supplied,
            differentiate directions directly and treat Wigner matrices as
            cached values. Otherwise differentiate the matrices themselves.

        Returns
        -------
        torch.Tensor
            Aggregated features, ``(num_nodes, irreps_out.dim)``, in ir_mul order.
        """
        return super().forward(
            features,
            radial,
            projection,
            wigner,
            cutoff,
            edge_index,
            num_nodes,
            vectors=vectors,
        )
