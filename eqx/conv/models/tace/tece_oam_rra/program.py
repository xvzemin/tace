"""Static expressions for the TECE-OAM-RRA interaction."""

from functools import lru_cache

from ....._metadata import parse_metadata
from ....program import Program, next_adjoint


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
    """Serialize the native O(2) paths and the checkpoint's normalization."""
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


@lru_cache(maxsize=64)
def build(metadata, radial_width, basis_width, angular_in, angular_out):
    """Build score and value expressions once for each static architecture."""
    (
        ell,
        mmax,
        channels,
        heads,
        hidden,
        gate_m0,
        ece,
        eps,
        attention_scale,
        temperature_min,
        temperature_max,
        scalar_act,
        tensor_act,
        linears,
        temperature_slot,
        parameter_sizes,
        update,
        radial_product,
    ) = parse_metadata(metadata)
    program = Program()
    n = ell + 1
    angular = sum((n - m) * (1 if m == 0 else 2) for m in range(mmax + 1))
    radial_channels = sum(n - m for m in range(mmax + 1)) * 2 * channels
    specs = [
        (angular_in * channels, "source"),
        (radial_width, "edge"),
        (radial_width * radial_channels, "shared"),
        (radial_channels, "shared"),
        (1, "edge"),
        (angular * angular_in, "edge"),
        (angular_out * angular, "edge"),
        (basis_width, "edge"),
    ]
    specs.extend((size, "shared") for size in parameter_sizes)
    values = [program.input(i, kind, size) for i, (size, kind) in enumerate(specs)]
    source = values[0]
    target = program.input(0, "target", angular_in * channels)
    paired = program.concatenate((source, target))
    paired = program.gather(
        paired,
        (
            side * angular_in * channels + a * channels + c
            for a in range(angular_in)
            for side in range(2)
            for c in range(channels)
        ),
    )
    local = program.matmul(values[5], paired, angular, angular_in, 2 * channels)
    descriptions = {entry[0]: entry[1:] for entry in linears}

    def linear(name, features):
        weight_slot, bias_slot, nw, nb, paths = descriptions[name]
        weight, bias = values[8 + weight_slot], values[8 + bias_slot]
        outputs, start_bias = [], 0
        for start, dim, ni, no, offset, scale in paths:
            matrix = program.scale(program.slice(weight, offset, ni * no), scale)
            result = program.matmul(
                program.slice(features, start, dim * ni), matrix, dim, ni, no
            )
            if dim == 1:
                result = program.binary(
                    "add", result, program.slice(bias, start_bias, no)
                )
                start_bias += no
            outputs.append(result)
        return program.concatenate(outputs)

    def activate(x, specification):
        op, scale = specification
        return program.scale(x if op == "identity" else program.unary(op, x), scale)

    source_local = program.gather(
        local, (a * 2 * channels + c for a in range(angular) for c in range(channels))
    )
    target_local = program.gather(
        local,
        (
            a * 2 * channels + channels + c
            for a in range(angular)
            for c in range(channels)
        ),
    )
    query, key = linear("query_proj", target_local), linear("key_proj", source_local)
    radial = linear("radial_proj", values[7])
    phase = program.scale(
        program.unary("tanh", program.slice(radial, heads, heads)), 3.141592653589793
    )
    orders, swap, sign = [], [], []
    offset = 0
    for m in range(mmax + 1):
        size = n - m
        for real in range(1 if m == 0 else 2):
            for degree in range(size):
                orders.append(m)
                swap.append(offset + (1 - real) * size + degree if m else degree)
                sign.append(-1 if real == 0 else 1)
        offset += size * (1 if m == 0 else 2)
    head_indices = tuple(
        c // (channels // heads) for _ in range(angular) for c in range(channels)
    )
    angle = program.gather(phase, head_indices)
    order_values = program.concatenate([program.constant(channels, m) for m in orders])
    angle = program.binary("mul", angle, order_values)
    rotated = program.binary(
        "add",
        program.binary("mul", key, program.unary("cos", angle)),
        program.binary(
            "mul",
            program.unary("sin", angle),
            program.binary(
                "mul",
                program.gather(
                    key, (a * channels + c for a in swap for c in range(channels))
                ),
                program.concatenate([program.constant(channels, s) for s in sign]),
            ),
        ),
    )
    score = program.scatter(program.binary("mul", query, rotated), head_indices, heads)
    temperature = program.binary(
        "add",
        program.constant(heads, temperature_min),
        program.scale(
            program.unary("sigmoid", values[8 + temperature_slot]),
            temperature_max - temperature_min,
        ),
    )
    score = program.binary(
        "add",
        program.scale(program.binary("mul", score, temperature), attention_scale),
        program.slice(radial, 0, heads),
    )
    weight = program.binary(
        "add",
        program.matmul(values[1], values[2], 1, radial_width, radial_channels),
        values[3],
    )
    empty = program.constant(1, 0)
    message = program.product((local, weight, empty), radial_product, 2)
    packed = linear("linear_up", message)
    width = hidden if ece else channels
    gates = sum((n if ece else n - m) for m in range(mmax + 1) if gate_m0 or m)
    gate = activate(program.slice(packed, 0, gates * width), tensor_act)
    scalar = activate(program.slice(packed, gates * width, n * width), scalar_act)
    if ece:
        coefs = activate(
            linear("linear_coefs", program.slice(message, 0, n * 2 * channels)),
            scalar_act,
        )
        message = program.product(
            (packed, packed, gate, coefs, scalar, empty), update, 5
        )
    else:
        features = program.slice(
            packed, gates * width, program.size(packed) - gates * width
        )
        # Order-zero gates are optional; positive orders share one scalar gate
        # between their two real coordinates.
        mapped, cursor = [], 0
        for m in range(mmax + 1):
            size = (n - m) * width
            if m or gate_m0:
                mapped.extend(range(cursor, cursor + size))
                if m:
                    mapped.extend(range(cursor, cursor + size))
                cursor += size
        tensors = (
            features
            if gate_m0
            else program.slice(features, n * width, program.size(features) - n * width)
        )
        tensors = program.binary("mul", tensors, program.gather(gate, mapped))
        message = tensors if gate_m0 else program.concatenate((scalar, tensors))
    message = linear("linear_down", message)
    value = program.matmul(values[6], message, angular_out, angular, channels)
    return tuple(program.nodes), score, value, tuple(specs), heads, channels, eps


@lru_cache(maxsize=64)
def first_adjoint(base, active):
    """Apply the weighted softmax adjoint before reversing the local update."""
    nodes, score, value, specs, heads, channels, _ = base
    program = Program(nodes)
    size = program.size(value)
    start = len(specs)
    result, denominator, maximum, grad_result, grad_denominator = [
        program.input(start + i, "target", width)
        for i, width in enumerate((size, heads, heads, size, heads))
    ]
    cutoff = program.input(4, "edge", 1)
    cutoff = program.gather(cutoff, (0,) * heads)
    exp = program.unary("exp", program.binary("add", score, program.scale(maximum, -1)))
    quotient = program.binary("mul", exp, program.unary("reciprocal", denominator))
    groups = tuple(
        c // (channels // heads)
        for _ in range(size // channels)
        for c in range(channels)
    )
    dot_value = program.scatter(
        program.binary("mul", value, grad_result), groups, heads
    )
    dot_result = program.scatter(
        program.binary("mul", result, grad_result), groups, heads
    )
    normalizer_grad = program.binary(
        "add",
        program.binary(
            "mul",
            quotient,
            program.binary(
                "add",
                program.binary("mul", cutoff, dot_value),
                program.scale(dot_result, -1),
            ),
        ),
        program.binary("mul", exp, grad_denominator),
    )
    grad_score = program.binary("mul", normalizer_grad, cutoff)
    grad_value = program.binary(
        "mul",
        grad_result,
        program.gather(
            program.binary("mul", quotient, program.binary("mul", cutoff, cutoff)),
            groups,
        ),
    )
    grad_cutoff = program.scatter(
        program.binary(
            "add",
            normalizer_grad,
            program.binary("mul", program.binary("mul", quotient, cutoff), dot_value),
        ),
        (0,) * heads,
        1,
    )
    original_cutoff = program.input(4, "edge", 1)
    outputs = program.adjoint(
        (score, value, original_cutoff), (grad_score, grad_value, grad_cutoff), active
    )
    return repr((tuple(program.nodes), outputs))
