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
        Per-edge values and radial derivatives, optionally batched over layers,
        evaluated at detached radii.
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
        if slope.ndim == 3:
            radial_gradient = radial_gradient.sum(0)
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


def supports_degree_jets(representation):
    """Whether fixed jn bases feed matching, unconditioned scalar MLP stages."""
    radial = representation.radial_basis
    if (
        radial.widths is None
        or type(radial.radial_fn).__name__ != "jnSphericalBesselBasis"
        or radial.apply_cutoff
        or radial.use_distance_transform
        or any(parameter.requires_grad for parameter in radial.parameters())
        or type(representation.edge_embedding).__name__
        not in ("ElementEdgeEmbedding", "Element2EdgeEmbedding")
        or any(
            type(update).__name__ != "IdentityEdgeUpdate"
            for update in representation.edge_updates
        )
    ):
        return False
    signatures = {}
    for layer in representation.interactions:
        network = layer.edge_info
        if not hasattr(network, "active") or not hasattr(network, "branches"):
            return False
        for degree, head in zip(network.active, network.branches):
            signature = []
            for stage in tuple(head.mlp)[:-1]:
                if type(stage).__name__ == "mlpLinear":
                    key = (
                        "linear",
                        stage.in_dim,
                        stage.out_dim,
                        stage.bias is not None,
                    )
                else:
                    function = getattr(stage, "activation", stage)
                    name = type(function).__name__
                    if (
                        name
                        not in (
                            "SiLU",
                            "ScaledSiLU",
                            "Tanh",
                            "ScaledTanh",
                            "Sigmoid",
                            "ScaledSigmoid",
                            "Identity",
                        )
                        or next(stage.parameters(), None) is not None
                    ):
                        return False
                    key = (name, getattr(stage, "scale_factor", 1.0))
                signature.append(key)
            signature = tuple(signature)
            if signatures.setdefault(degree, signature) != signature:
                return False
    return True


def jn_slope(radial, radius):
    """Evaluate the fixed jn basis derivative at the current detached geometry."""
    from tace.models.radial import SphericalBesselJn

    basis = radial.radial_fn
    frequency = basis.zeros.to(torch.float64) / basis.cutoff.to(torch.float64)
    arguments = basis.zeros.to(torch.float64) * (
        radius.detach().to(torch.float64) / basis.cutoff.to(torch.float64)
    )
    derivative = torch.cat(
        [
            SphericalBesselJn.apply(order, 1, value, parameters)
            for order, value, parameters in zip(
                basis.order,
                arguments.split(basis.num_zero, -1),
                basis.bessel_parameters,
            )
        ],
        -1,
    )
    return derivative * basis.normalizer.to(torch.float64) * frequency


def packed_hidden_jets(networks, value, slope, *, batched=False):
    """Evaluate hidden values and radial slopes in the same batched GEMMs."""
    sequences = [tuple(network.mlp)[:-1] for network in networks]
    if not sequences[0]:
        if batched:
            return (
                value.unsqueeze(0).expand(len(networks), -1, -1),
                slope.unsqueeze(0).expand(len(networks), -1, -1),
            )
        return (value,) * len(networks), (slope,) * len(networks)
    rows = value.shape[0]
    pair = torch.cat((value, slope), 0).unsqueeze(0)
    for stage in range(len(sequences[0])):
        layers = [sequence[stage] for sequence in sequences]
        layer = layers[0]
        if type(layer).__name__ == "mlpLinear":
            pair = torch.matmul(
                pair, torch.stack([item.get_weight() for item in layers])
            )
            if layer.bias is not None:
                value, slope = pair.unflatten(1, (2, rows)).unbind(1)
                value = value + torch.stack([item.bias for item in layers])[:, None]
                pair = torch.cat((value, slope), 1)
        elif not isinstance(layer, torch.nn.Identity):
            value, slope = pair.unflatten(1, (2, rows)).unbind(1)
            pair = torch.cat((layer(value), slope * activation_slope(layer, value)), 1)
    value, slope = pair.unflatten(1, (2, rows)).unbind(1)
    if batched:
        return value, slope
    return value.unbind(0), slope.unbind(0)


def hidden_jets(representation, networks, radius, node_type, edge_index):
    """Evaluate analytic scalar-radius JVPs without dual-tensor shape guards."""
    embedding = representation.edge_embedding
    length = radius.detach()
    basis_fn = representation.radial_basis
    basis, _ = basis_fn(length, node_type, edge_index, representation.atomic_numbers)
    basis = basis.to(representation.model_dtype)
    slope = bessel_slope(basis_fn, length).to(basis)
    if type(embedding).__name__ == "LinearEdgeEmbedding":
        projection = embedding.radial_proj
        weight = projection.scalar_matrix()
        radial, radial_slope = basis @ weight, slope @ weight
        if projection.bias is not None:
            radial = radial + projection.bias
    elif not hasattr(embedding, "radial_proj"):
        radial, radial_slope = basis, slope
    else:
        raise ValueError("Radial jets require identity or linear radial embedding")
    sequences = [tuple(network.mlp)[:-1] for network in networks]
    if not sequences or any(len(layers) != len(sequences[0]) for layers in sequences):
        raise ValueError("Radial jets require matching hidden MLP stages")
    updates = representation.edge_updates
    source_types, target_types = node_type[edge_index[0]], node_type[edge_index[1]]
    if not sequences[0]:
        values = [
            torch.cat(
                (
                    radial,
                    update.target_embedding(target_types),
                    update.source_embedding(source_types),
                ),
                -1,
            )
            if hasattr(update, "source_embedding")
            else radial
            for update in updates
        ]
        slopes = [
            torch.nn.functional.pad(
                radial_slope, (0, value.shape[-1] - radial_slope.shape[-1])
            )
            for value in values
        ]
        return values, slopes
    first = [layers[0] for layers in sequences]
    if any(type(layer).__name__ != "mlpLinear" for layer in first):
        raise ValueError("Radial jets require a linear first hidden stage")
    weights = torch.stack([layer.get_weight() for layer in first])
    width = radial.shape[-1]
    radial_weight = weights[:, :width]
    values = torch.matmul(radial.unsqueeze(0), radial_weight)
    tangents = torch.matmul(radial_slope.unsqueeze(0), radial_weight)
    if first[0].bias is not None:
        values = values + torch.stack([layer.bias for layer in first])[:, None, :]
    if hasattr(updates[0], "source_embedding"):
        target_embedding = torch.stack(
            [update.target_embedding.weight for update in updates]
        )
        source_embedding = torch.stack(
            [update.source_embedding.weight for update in updates]
        )
        split = width + target_embedding.shape[-1]
        values = (
            values
            + torch.bmm(target_embedding, weights[:, width:split])[:, target_types]
        )
        values = (
            values + torch.bmm(source_embedding, weights[:, split:])[:, source_types]
        )
    for stage in range(1, len(sequences[0])):
        layers = [sequence[stage] for sequence in sequences]
        layer = layers[0]
        if any(type(item) is not type(layer) for item in layers):
            raise ValueError("Radial jets require matching hidden stage types")
        if type(layer).__name__ == "mlpLinear":
            weights = torch.stack([item.get_weight() for item in layers])
            values, tangents = torch.bmm(values, weights), torch.bmm(tangents, weights)
            if layer.bias is not None:
                values = (
                    values + torch.stack([item.bias for item in layers])[:, None, :]
                )
        elif isinstance(layer, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            normalized = [
                torch.func.jvp(item, (x,), (dx,))
                for item, x, dx in zip(layers, values.unbind(0), tangents.unbind(0))
            ]
            values, tangents = (torch.stack(part) for part in zip(*normalized))
        else:
            if any(tuple(item.parameters()) for item in layers):
                raise ValueError("Parameterized radial jet activations are unsupported")
            tangents = tangents * activation_slope(layer, values)
            values = layer(values)
    return tuple(values.unbind(0)), tuple(tangents.unbind(0))
