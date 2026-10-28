"""Static expressions for the TECE-OAM-RRA interaction."""

from ....utils.metadata import parse_metadata


def activation(module):
    """Describe supported pointwise activations without capturing the module."""
    name = type(module).__name__
    if name == "ScaledActivation":
        inner = activation(module.activation)
        return None if inner is None else (inner[0], inner[1] * module.scale_factor)
    names = {
        "Identity": "identity",
        "SiLU": "silu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
    }
    name = name.removeprefix("Scaled")
    return (
        (names[name], float(getattr(module, "scale_factor", 1)))
        if name in names
        else None
    )


def metadata(module):
    """Serialize O(2) paths and their normalization."""
    scalar, tensor = activation(module.scalar_act), activation(module.tensor_act)
    if scalar is None or tensor is None or not module.use_radial_rotary_attention:
        return None
    names = tuple(name for name, _ in module.named_parameters())
    linears = []
    for name in (
        "linear_up",
        "linear_down",
        "linear_coefs",
        "query_proj",
        "key_proj",
        "radial_proj",
    ):
        linear = getattr(module, name, None)
        if linear is None:
            continue
        paths = tuple(
            (
                linear._input_slices[ins.i_in].start,
                linear.irreps_out[ins.i_out].ir.dim,
                *ins.path_shape,
                offset,
                ins.path_weight,
            )
            for ins, (offset, _) in zip(
                linear._weight_instructions, linear._weight_offsets
            )
        )
        linears.append(
            (
                name,
                names.index(name + ".weight"),
                names.index(name + ".bias"),
                linear.weight_numel,
                linear.bias_numel,
                paths,
            )
        )
    return repr(
        (
            module.lmax,
            module.mmax,
            module.num_channel,
            module.num_head,
            module.edge_ace_hidden,
            module.gate_m0,
            module.use_asymmetric_contraction,
            module.attention_eps,
            module.attention_scale,
            module.temperature_min,
            module.temperature_max,
            scalar,
            tensor,
            tuple(linears),
            names.index("temperature_logit"),
            tuple(p.numel() for p in module.parameters()),
            parse_metadata(module._eqx_update_metadata)
            if module.use_asymmetric_contraction
            else (),
            parse_metadata(module._eqx_weight_metadata),
        )
    )
