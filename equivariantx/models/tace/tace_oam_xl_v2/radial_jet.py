"""Training-only scalar radial jets with exact energy/force parameter gradients.

At the current geometry r0, F(theta, r) is represented by its value and first
radial derivative. This preserves F, dF/dr and d²F/(dtheta dr) at r0. It does
not preserve d²F/dr², and must not be used for Hessian/phonon objectives or
coordinate optimization through predicted forces.
"""

import torch


class AttachRadialJet(torch.autograd.Function):
    """Attach an independently evaluated radial slope to a parameter graph.

    Parameters
    ----------
    value, slope : torch.Tensor
        Per-edge values and radial derivatives, evaluated at detached radii.
        Both retain their complete parameter dependence.
    radius : torch.Tensor
        Live per-edge radii of shape (edges, 1).
    """

    @staticmethod
    def forward(value, slope, radius):
        return value.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, slope, radius = inputs
        ctx.save_for_backward(slope)
        ctx.radius_dtype = radius.dtype

    @staticmethod
    def backward(ctx, gradient):
        (slope,) = ctx.saved_tensors
        radial_gradient = (gradient * slope).sum(-1, keepdim=True)
        return gradient, None, radial_gradient.to(ctx.radius_dtype)


def bessel_slope(radial_basis, radius):
    """Differentiate the current uncut j0 basis, including its origin limit."""
    if (
        type(radial_basis.radial_fn).__name__ != "j0SphericalBesselBasis"
        or radial_basis.use_distance_transform
        or radial_basis.apply_cutoff
    ):
        raise ValueError(
            "Radial jets currently require the untransformed, uncut j0 basis"
        )
    basis = radial_basis.radial_fn
    z = radius * basis.bessel_weights
    small = z.abs() < 1e-3
    safe = torch.where(small, torch.ones_like(z), z)
    regular = (safe * safe.cos() - safe.sin()) / safe.square()
    series = z * (
        -1 / 3 + z.square() * (1 / 30 + z.square() * (-1 / 840 + z.square() / 45360))
    )
    return (
        basis.prefactor
        * basis.bessel_weights.square()
        * torch.where(small, series, regular)
    )


def activation_slope(layer, value):
    """Use analytic slopes of the architecture's scaled scalar activations."""
    scale = getattr(layer, "scale_factor", 1.0)
    function = getattr(layer, "activation", layer)
    name = type(function).__name__
    if name in ("SiLU", "ScaledSiLU"):
        sigmoid = value.sigmoid()
        return scale * sigmoid * (1 + value * (1 - sigmoid))
    if name in ("Tanh", "ScaledTanh"):
        return scale * (1 - value.tanh().square())
    if name in ("Sigmoid", "ScaledSigmoid"):
        sigmoid = value.sigmoid()
        return scale * sigmoid * (1 - sigmoid)
    if name == "Identity":
        return torch.ones_like(value) * scale
    raise ValueError(f"Unsupported radial jet activation: {type(layer).__name__}")


def hidden_jets(representation, radius, node_type, edge_index):
    """Evaluate analytic scalar-radius JVPs without dual-tensor shape guards."""
    embedding = representation.edge_embedding
    if type(embedding).__name__ != "LinearEdgeEmbedding":
        raise ValueError("Radial jets currently require LinearEdgeEmbedding")
    projection = embedding.radial_proj
    if len(projection.weight) != 1 or len(projection.linear.instructions) != 1:
        raise ValueError("Expected one scalar radial embedding matrix")
    length = radius.detach()
    basis, _ = representation.radial_basis(
        length, node_type, edge_index, representation.atomic_numbers
    )
    basis = basis.to(representation.model_dtype)
    slope = bessel_slope(representation.radial_basis, length).to(basis)
    weight = projection.weight[0] * projection.linear.instructions[0].path_weight
    radial, radial_slope = basis @ weight, slope @ weight
    if projection.bias is not None:
        radial = radial + projection.bias
    sequences = [
        tuple(layer.edge_info.mlp)[:-1] for layer in representation.interactions
    ]
    if not sequences or any(len(layers) != len(sequences[0]) for layers in sequences):
        raise ValueError("Radial jets require matching hidden MLP stages")
    first = [layers[0] for layers in sequences]
    if any(type(layer).__name__ != "mlpLinear" for layer in first):
        raise ValueError("Radial jets require a linear first hidden stage")
    weights = torch.stack([layer.get_weight() for layer in first])
    updates = representation.edge_updates
    target_embedding = torch.stack(
        [update.target_embedding.weight for update in updates]
    )
    source_embedding = torch.stack(
        [update.source_embedding.weight for update in updates]
    )
    width, elements = radial.shape[-1], target_embedding.shape[-1]
    radial_weight = weights[:, :width]
    values = torch.matmul(radial.unsqueeze(0), radial_weight)
    tangents = torch.matmul(radial_slope.unsqueeze(0), radial_weight)
    if first[0].bias is not None:
        values = values + torch.stack([layer.bias for layer in first])[:, None, :]
    source_types, target_types = node_type[edge_index[0]], node_type[edge_index[1]]
    values = (
        values
        + torch.bmm(target_embedding, weights[:, width : width + elements])[
            :, target_types
        ]
    )
    values = (
        values
        + torch.bmm(source_embedding, weights[:, width + elements :])[:, source_types]
    )
    for stage in range(1, len(sequences[0])):
        layers = [sequence[stage] for sequence in sequences]
        layer = layers[0]
        if any(type(item) != type(layer) for item in layers):
            raise ValueError("Radial jets require matching hidden stage types")
        if type(layer).__name__ == "mlpLinear":
            weights = torch.stack([item.get_weight() for item in layers])
            values, tangents = torch.bmm(values, weights), torch.bmm(tangents, weights)
            if layer.bias is not None:
                values = (
                    values + torch.stack([item.bias for item in layers])[:, None, :]
                )
        else:
            if any(tuple(item.parameters()) for item in layers):
                raise ValueError("Parameterized radial jet activations are unsupported")
            tangents = tangents * activation_slope(layer, values)
            values = layer(values)
    return tuple(values.unbind(0)), tuple(tangents.unbind(0))
