import math

import torch
from e3nn import o3

from eqx import o2
from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..linear import IndexedFeatures
from ..mlp import ACTIVATION, MLP_TYPES


class ConditionalO2Weights(torch.nn.Module):
    """Generate O(2) kernel weights from channelwise query-key products.

    Products pair the same irrep copy and channel in query and key features.
    Radial coefficients are independent for every copy and channel.
    All weights are time-even.

    Parameters
    ----------
    linear : eqx.o2.ConditionalLinear
        Conditional map defining the feature layout and weight counts.
    in_features : int
        Number of invariant edge features.
    hidden : list of int
        Hidden widths of the radial MLP.
    bias : bool
        Include biases in the radial MLP.
    mlp_type : str
        MLP implementation registered in ``MLP_TYPES``.
    normalization : str or None
        Radial MLP normalization: ``layer_norm``, ``rms_norm``, or None.
    weights_0o_act : {"tanh", "softsign", "identity", None}, optional
        Odd activation applied to ``weights_0o`` after radial modulation.
        Defaults to ``"tanh"``. ``None`` leaves the weights unchanged.
    """

    def __init__(
        self,
        linear: o2.ConditionalLinear,
        in_features: int,
        hidden: list[int],
        bias: bool,
        mlp_type: str,
        normalization: str | None,
        weights_0o_act: str | None = "tanh",
    ) -> None:
        super().__init__()
        if weights_0o_act not in (None, "tanh", "softsign", "identity"):
            raise ValueError(
                "weights_0o_act must preserve odd parity: use 'tanh', 'softsign', "
                "'identity', or None."
            )
        self.weights_0o_act = ACTIVATION[weights_0o_act]()
        self.irreps_in = linear.irreps_in
        self.input_sizes = tuple(entry.dim for entry in self.irreps_in)
        self.weight_numel_0e = linear.weight_numel_0e
        self.weight_numel_0o = linear.weight_numel_0o
        self.weight_numel_2m = linear.weight_numel_2m
        self.coefficient_numel_2m = linear.weight_numel_0o
        self.mlp = MLP_TYPES[mlp_type](
            [
                in_features,
                *hidden,
                self.weight_numel_0e + self.weight_numel_0o + self.coefficient_numel_2m,
            ],
            bias=bias,
            layer_norm=normalization == "layer_norm",
            rms_norm=normalization == "rms_norm",
            act="silu",
        )
        self.num_layers = self.mlp.num_layers

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        edge_feats: torch.Tensor | IndexedFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct the external kernel weights.

        Parameters
        ----------
        query, key : torch.Tensor
            Projected local features with shape ``(num_edges, irreps_in.dim)``.
        edge_feats : torch.Tensor or IndexedFeatures
            Invariant edge inputs to the MLP.

        Returns
        -------
        tuple of torch.Tensor
            ``weights_0e``, ``weights_0o``, and ``weights_2m``, with shapes
            ``(num_edges, weight_numel_0e)``, ``(num_edges, weight_numel_0o)``,
            and ``(num_edges, weight_numel_2m)``.
        """
        queries = query.split(self.input_sizes, -1)
        keys = key.split(self.input_sizes, -1)
        even, odd, positive, negative = [], [], [], []
        for (ir, mul), u, v in zip(self.irreps_in, queries, keys):
            shape = (query.shape[0], ir.dim, mul)
            u, v = u.reshape(shape), v.reshape(shape)
            even.append((u * v).sum(1) / math.sqrt(ir.dim))
            if ir.m == 0:
                continue
            up, um = u.unbind(1)
            vp, vm = v.unbind(1)
            odd.append((up * vm - um * vp) / math.sqrt(2))
            positive.append((up * vp - um * vm) / math.sqrt(2))
            negative.append((up * vm + um * vp) / math.sqrt(2))
        scalars = self.mlp(edge_feats)
        even_coefficients, odd_coefficients, tensor_coefficients = scalars.split(
            (self.weight_numel_0e, self.weight_numel_0o, self.coefficient_numel_2m), -1
        )
        weights_0e = even_coefficients * torch.cat(even, -1) if even else scalars[:, :0]
        weights_0o = odd_coefficients * torch.cat(odd, -1) if odd else scalars[:, :0]
        weights_2m = (
            torch.cat(
                (
                    tensor_coefficients * torch.cat(positive, -1),
                    tensor_coefficients * torch.cat(negative, -1),
                ),
                dim=-1,
            )
            if positive
            else scalars[:, :0]
        )
        return weights_0e, self.weights_0o_act(weights_0o), weights_2m


class ConditionalO2ScatterTensorProduct(torch.nn.Module):
    """Project local queries, keys and values for an O(2) convolution.

    Parameters
    ----------
    irreps_in, irreps_out : e3nn.o3.Irreps or str
        Node input and output representations in flattened ``mul_ir`` layout.
    mmax : int
        Largest retained local order.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        mmax: int,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.local_frame_in = o2.LocalFrame(self.irreps_in, mmax=mmax)
        self.local_frame_out = o2.LocalFrame(self.irreps_out, mmax=mmax, reverse=True)
        self.reshape_in = LayoutTransform(
            self.irreps_in, layout_in="flatten_mul_ir", layout_out="flatten_ir_mul"
        )
        self.reshape_out = LayoutTransform(
            self.irreps_out, layout_in="flatten_mul_ir", layout_out="flatten_ir_mul"
        )
        irreps_local_in = self.local_frame_in.irreps_out
        irreps_local_out = self.local_frame_out.irreps_out
        self.q_proj = o2.Linear(irreps_local_in, irreps_local_out)
        self.k_proj = o2.Linear(irreps_local_in, irreps_local_out)
        self.v_proj = o2.Linear(irreps_local_in, irreps_local_out)
        self.linear = o2.ConditionalLinear(irreps_local_out)
        self.weight_numel = self.linear.weight_numel

    def to_local(self, node_feats, edge_index, edge_wigner):
        """Gather and rotate both endpoints into the same edge frame."""
        features = self.reshape_in(node_feats)
        paired = self.local_frame_in.to_local(features[edge_index.T], edge_wigner)
        return paired[:, 0], paired[:, 1]

    def forward_local(
        self,
        features,
        weights_0e,
        weights_0o,
        weights_2m,
        edge_index,
        edge_wigner_inv,
        num_nodes,
        edge_cutoff=None,
    ):
        """Apply the map to projected values and aggregate global outputs."""
        message = self.linear(features, weights_0e, weights_0o, weights_2m)
        message = self.local_frame_out.to_global(message, edge_wigner_inv)
        if edge_cutoff is not None:
            message = message * edge_cutoff
        message = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
        return self.reshape_out.inverse(message)

    def forward(
        self,
        node_feats,
        weights_0e,
        weights_0o,
        weights_2m,
        edge_index,
        edge_wigner,
        edge_wigner_inv,
        edge_cutoff=None,
    ):
        """Evaluate a convolution with externally constructed tensor weights.

        Parameters
        ----------
        node_feats : torch.Tensor
            Node features with shape ``(num_nodes, irreps_in.dim)``.
        weights_0e, weights_0o, weights_2m : torch.Tensor
            Per-edge weights following :class:`eqx.o2.ConditionalLinear`.
        edge_index : torch.Tensor
            Source and target indices with shape ``(2, num_edges)``.
        edge_wigner, edge_wigner_inv : torch.Tensor
            Forward and inverse frame rotations.
        edge_cutoff : torch.Tensor, optional
            Message envelope with shape ``(num_edges, 1)``.

        Returns
        -------
        torch.Tensor
            Aggregated features with shape ``(num_nodes, irreps_out.dim)``.
        """
        source = self.local_frame_in.to_local(
            self.reshape_in(node_feats)[edge_index[0]], edge_wigner
        )
        return self.forward_local(
            self.v_proj(source),
            weights_0e,
            weights_0o,
            weights_2m,
            edge_index,
            edge_wigner_inv,
            node_feats.shape[0],
            edge_cutoff,
        )
