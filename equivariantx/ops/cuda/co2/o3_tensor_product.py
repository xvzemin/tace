"""Native transverse kernels with compact angular and channel scheduling."""

from ...torch.co2 import o3_tensor_product as torch_kernel


def forward(module, features, vectors, weight):
    """Keep native sparse angle kernels visible to joint AOT compilation."""
    return torch_kernel.forward(module, features, vectors, weight)


def scatter(module, features, vectors, edge_index, weight, num_nodes):
    """Aggregate compact channels before expanding Cartesian outputs."""
    return torch_kernel.scatter(
        module, features, vectors, edge_index, weight, num_nodes
    )


def reference(module, features, vectors, weight):
    return torch_kernel.reference(module, features, vectors, weight)


def contract(module, features, vectors, weight):
    return torch_kernel.unprojected(module, features, vectors, weight)
