"""Node--edge--node convolutions with Wigner-6j recoupling."""

from .tensor_product import Wigner6jTensorProduct


class Wigner6jTensorProductConv(Wigner6jTensorProduct):
    """Recouple node inputs before the edge product and destination reduction.

    Parameters
    ----------
    irreps_in1, irreps_in2, irreps_in3 : e3nn.o3.Irreps or str
        First-node, edge, and third-node representations in ``mul_ir`` layout.
    irreps_mid, irreps_out : e3nn.o3.Irreps or str
        Intermediate and output representations of the standard coupling tree.
    instructions1, instructions2 : list of tuple
        Instructions for the two standard ``uvu`` tensor products.
    irrep_normalization, path_normalization : str, optional
        Normalizations of the standard tensor products.
    internal_weights, shared_weights : bool, optional
        Store internal weights and share weights between edges.
    backend : {"cuda", "torch"}, optional
        CUDA or PyTorch evaluation. Defaults to "cuda".
    """

    def forward(
        self,
        x1,
        x2,
        x3,
        edge_index,
        weight1=None,
        weight2=None,
        num_nodes=None,
        *,
        backend=None,
    ):
        """Sum recoupled edge messages at destination nodes.

        Parameters
        ----------
        x1, x3 : torch.Tensor
            Node features of shape ``(nodes, irreps_inN.dim)``. Both are
            gathered at the source node in the standard reference.
        x2 : torch.Tensor
            Edge features of shape ``(edges, irreps_in2.dim)``.
        edge_index : torch.Tensor
            Source and target indices of shape ``(2, edges)``.
        weight1, weight2 : torch.Tensor, optional
            Standard tensor-product weights, shared or one vector per edge.
        num_nodes : int, optional
            Destination-node count. Defaults to the first input's node count.
        backend : {"cuda", "torch"}, optional
            Override the construction-time backend for this call.

        Returns
        -------
        torch.Tensor
            Destination features of shape ``(num_nodes, irreps_out.dim)``.
        """
        if x1.ndim != 2 or x2.ndim != 2 or x3.ndim != 2:
            raise ValueError("Convolution inputs must be matrices.")
        if x1.shape[0] != x3.shape[0] or edge_index.shape != (2, x2.shape[0]):
            raise ValueError("Node and edge counts must match their inputs.")
        for value, width in zip((x1, x2, x3), self.input_dims):
            if value.shape[-1] != width:
                raise ValueError("Input widths must match their irreps.")
        weight1, weight2 = self.get_weights(weight1, weight2)
        weight = self.combine_weights(weight1, weight2)
        if weight.ndim > 2 or (
            weight.ndim == 2 and weight.shape[0] not in (1, x2.shape[0])
        ):
            raise ValueError("Weights must be shared or have one row per edge.")
        backend = self.backend if backend is None else backend
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be cuda or torch.")
        intermediate = self.couple_inputs(x1, x3, backend)
        num_nodes = x1.shape[0] if num_nodes is None else num_nodes
        if backend == "cuda" and x1.is_cuda:
            return self.conv132(
                intermediate,
                x2.index_select(-1, self.input2_index),
                weight.unsqueeze(0) if weight.ndim == 1 else weight,
                weight.new_empty((0, self.tp132.weight_numel)),
                edge_index,
                num_nodes,
            ).index_select(-1, self.output_index)
        message = self.tp132(intermediate[edge_index[0]], x2, weight)
        return message.new_zeros((num_nodes, self.output_dim)).index_add(
            0, edge_index[1], message
        )

    def forward_reference(
        self, x1, x2, x3, edge_index, weight1=None, weight2=None, num_nodes=None
    ):
        """Evaluate the standard edge-level tree and sum at destinations."""
        message = super().forward_reference(
            x1[edge_index[0]], x2, x3[edge_index[0]], weight1, weight2
        )
        num_nodes = x1.shape[0] if num_nodes is None else num_nodes
        return message.new_zeros((num_nodes, self.output_dim)).index_add(
            0, edge_index[1], message
        )
