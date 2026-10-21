"""Matrix tensor products evaluated in bounded edge tiles."""

from ..program import Program


def matrix_program(
    tensor_product,
    input_storage="edge",
    weight_storage="edge",
    output_storage="edge",
    cutoff=False,
):
    """Build a matrix product expression and its shared-buffer layout."""
    tp = tensor_product
    program = Program()
    x = program.input(0, input_storage, tp.input_dim1)
    y = program.input(1, "edge", tp.input_dim2)
    weight = program.input(2, weight_storage, tp.weight_numel)
    buffers = []

    def basis(name):
        slot = 3 + len(buffers)
        buffers.append(name)
        return program.input(slot, "shared", tp.constants[name].numel())

    def indices(name):
        return tuple(tp.constants[name].tolist())

    def factors(value, name):
        return program.binary(
            "mul", value, program.gather(weight, indices(f"{name}_weight_index"))
        )

    width, size = tp.num_channels, tp.matrix_dim
    inputs = []
    for value, entries in zip((x, y), tp.inputs):
        matrices = []
        for name, mul, dim in entries:
            features = factors(program.gather(value, indices(f"{name}_index")), name)
            matrix = program.matmul(
                features, basis(f"{name}_scaled_basis"), mul, dim, size * size
            )
            if mul == 1 and width != 1:
                matrix = program.gather(matrix, tuple(range(size * size)) * width)
            matrices.append(program.transpose(matrix, width, size * size))
        inputs.append(matrices)
    paths = tuple(
        ((a * size + k, k * size + b, a * size + b), 1.0)
        for a in range(size)
        for b in range(size)
        for k in range(size)
    )
    metadata = (width, (size * size,) * 3, paths)
    values = []
    for i, j, name, dim in tp.routes:
        matrix = program.product(
            (inputs[0][i], inputs[1][j], program.constant(width * size * size, 0)),
            metadata,
            2,
        )
        matrix = program.transpose(matrix, size * size, width)
        value = program.matmul(
            matrix, basis(f"{name}_scaled_basis"), width, size * size, dim
        )
        values.append(
            program.scatter(
                factors(value, name), indices(f"{name}_index"), tp.output_dim
            )
        )
    result = program.sum(values, tp.output_dim)
    if cutoff:
        envelope = program.input(3 + len(buffers), "edge", 1)
        result = program.binary(
            "mul", result, program.gather(envelope, (0,) * tp.output_dim)
        )
    return repr((tuple(program.nodes), ((result, 0, output_storage),))), tuple(buffers)
