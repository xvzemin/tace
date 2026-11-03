"""Output allocations shared by operator tracing and CUDA kernels."""

import torch

from .metadata import decode_plans, parse_metadata

NODE, MATRIX, EDGE = 0, 1, 2


def contraction_outputs(program, operands):
    program = parse_metadata(program)
    results = [None] * (1 + max(slot for _, _, pairs in program for _, slot in pairs))
    for mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


def direction_outputs(program, operands):
    program = parse_metadata(program)
    results = [None] * (
        1 + max(slot for _, _, _, pairs in program for _, slot in pairs)
    )
    for _, mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


def bilinear_outputs(metadata, program, node_type, operands):
    terms = parse_metadata(program)
    outputs = [None] * (max(slot for _, _, _, slot in terms) + 1)
    for plan, mapping, role, slot in terms:
        if outputs[slot] is None:
            rows = operands[mapping[2]].shape[0] if role == 2 else node_type.shape[0]
            outputs[slot] = operands[0].new_empty(
                (rows, decode_plans(metadata)[plan][0][role])
            )
    return outputs


def channel_product_outputs(metadata, program, operands):
    width, dims, _ = parse_metadata(metadata)
    dims = tuple(d * (width or 1) for d in dims)
    terms = parse_metadata(program)
    outputs = [None] * (max(slot for _, _, slot, _ in terms) + 1)
    for mapping, role, slot, _ in terms:
        if outputs[slot] is None:
            rows = operands[mapping[role]].size(0)
            outputs[slot] = operands[0].new_empty(
                (1 if rows == 1 else operands[0].shape[0], dims[role])
            )
    return outputs


def edge_outputs(metadata, inputs, source, target, num_nodes):
    nodes, outputs = parse_metadata(metadata)
    specifications = {slot: (nodes[root][1], kind) for root, slot, kind in outputs}
    return [
        inputs[0].new_empty(
            (
                1
                if kind == "shared"
                else source.numel()
                if kind == "edge"
                else num_nodes
            ),
            width,
        )
        for _, (width, kind) in sorted(specifications.items())
    ]


def tece_outputs(metadata, source, target, inputs):
    description = parse_metadata(metadata)
    if len(description) == 2:
        description = parse_metadata(description[0])
    channels, heads = description[2:4]
    nodes = inputs[0].shape[0]
    return [
        inputs[0].new_empty(
            (
                nodes,
                (description[0] + 1) ** 2
                if inputs[5].ndim == 2
                else inputs[6].shape[1],
                channels,
            )
        ),
        inputs[0].new_empty((nodes, heads)),
        inputs[0].new_empty((nodes, heads)),
    ]


def tece_contraction_outputs(metadata, source, target, inputs):
    slots = sorted({slot for _, slot, _ in parse_metadata(metadata)[1]})
    return [
        torch.empty_like(inputs[slot], memory_format=torch.contiguous_format)
        for slot in slots
    ]


def linear_output(metadata, role, node_type, features, weight, output):
    din, dout = parse_metadata(metadata)[:2]
    shape = (
        weight.shape if role == 1 else (node_type.shape[0], din if role == 0 else dout)
    )
    return features.new_empty(shape)


def ace_outputs(metadata, program, node_type, operands):
    dims, weighted, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (max(slot for _, _, slot in terms) + 1)
    for mapping, role, slot in terms:
        if outputs[slot] is None:
            rows = (
                operands[mapping[2]].shape[0]
                if weighted and role == 2
                else node_type.shape[0]
            )
            outputs[slot] = operands[0].new_empty((rows, dims[role]))
    return outputs


def symmetric_outputs(metadata, program, node_type, operands):
    input_dim, paths, output_dim, degree, _ = parse_metadata(metadata)
    terms = parse_metadata(program)
    outputs = [None] * (max(term[2] for term in terms) + 1)
    for mapping, role, slot, _ in terms:
        if outputs[slot] is None:
            size = (
                paths
                if role == degree
                else output_dim
                if role == degree + 1
                else input_dim
            )
            nodes = (
                operands[mapping[role]].shape[0]
                if role == degree
                else node_type.shape[0]
            )
            outputs[slot] = operands[0].new_empty((nodes, size, operands[0].shape[-1]))
    return outputs


def rotation_output(features, matrix, edge, metadata, role):
    if role == 0:
        return torch.empty_like(features)
    if role == 1:
        return torch.empty_like(matrix)
    return features.new_empty((features.shape[0], matrix.shape[1], features.shape[2]))


def quaternion_polynomial_output(lmax, transpose, values):
    width = 4 if transpose else sum((2 * l + 1) ** 2 for l in range(lmax + 1))
    return values[0].new_empty((values[0].size(0), width))


def gather_sum_output(features, indices):
    rows = features[0].size(0) if indices[0] is None else indices[0].numel()
    return features[0].new_empty((rows, features[0].size(1)))


def grouped_permute_output(features, indices, inverse, widths, transpose=False):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


def indexed_sum_output(
    features, indices, ptr, transpose_indices, transpose_ptr, count, transpose=False
):
    return features.new_empty((features.size(0), ptr.numel() - 1))


def alignment_outputs(key, values):
    from .scalar import alignment_program

    dtype = "float" if values[0].dtype == torch.float32 else "double"
    _, outputs, _ = alignment_program(parse_metadata(key), dtype)
    return [values[0].new_empty((values[0].size(0), len(group))) for group in outputs]


def wigner_rotation_output(cg, output, values):
    return torch.empty_like(values[output], memory_format=torch.contiguous_format)


def packed_wigner_output(vectors, degree, coefficients):
    width = sum((2 * l + 1) ** 2 for l in range(degree + 1))
    return vectors.new_empty(vectors.size(0), width)


def attention_adjoint_outputs(values, grad, weight, target, num_nodes):
    return (
        torch.empty_like(values, memory_format=torch.contiguous_format),
        torch.empty_like(weight, memory_format=torch.contiguous_format),
        values.new_empty((num_nodes, values.shape[1])),
    )


def indexed_frame_output(nodes, matrix, edges, index, num_nodes, layout, role):
    ni, no, nw, _ = parse_metadata(layout)
    shape = ((num_nodes, ni), (index.numel(), nw), (index.numel(), no))[role]
    return nodes.new_empty(shape)


def rotary_output(query, key, phase, score, lmax, mmax, heads, role):
    if role < 2:
        return torch.empty_like(query)
    if role == 2:
        return torch.empty_like(phase)
    return query.new_empty((query.shape[0], heads))


def o3_outputs(metadata, program, source, target, operands, *, batch_paths=False):
    return contraction_outputs(program, operands)
