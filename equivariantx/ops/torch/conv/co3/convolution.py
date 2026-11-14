"""Cartesian layout, geometry and compact coefficient execution."""

import torch

from .... import config


def evaluate(
    module,
    features,
    edge_attrs,
    radial,
    projection,
    edge_index,
    num_nodes,
    vectors,
    amplitudes,
    radial_network,
):
    """Evaluate validated Cartesian operands."""
    from equivariantx.o3.cgtp.convolution import O3TensorProductConv

    if module.input_basis == "spherical":
        features = module.to_cartesian(features)
    if module.symmetric_inputs:
        values = features.new_zeros((features.size(0), module.symmetric_dim)).index_add(
            -1, module.symmetric_index, features
        )
        features = (values / module.symmetric_count).index_select(
            -1, module.symmetric_index
        )
    if vectors is not None:
        vectors = (
            torch.nn.functional.normalize(vectors, dim=-1)
            if module.normalize
            else vectors
        )
        attrs, offset = ([], 0)
        for mul, ir in module.irreps_in2:
            harmonic = module.harmonics[str(ir.l)](vectors)
            amplitude = (
                features.new_ones((1, mul))
                if amplitudes is None
                else amplitudes.expand(amplitudes.size(0), module.amplitude_dim)[
                    :, offset : offset + mul
                ]
            )
            attrs.append((amplitude.unsqueeze(-1) * harmonic.unsqueeze(-2)).flatten(1))
            offset += mul
        edge_attrs = torch.cat(attrs, dim=-1) if attrs else vectors[:, :0]
        edge_attrs = edge_attrs + vectors.sum() * 0
        vectors = None
    if vectors is not None and module.pack_inputs:
        from equivariantx.ops.common.layout import indexed_sum

        if features.size(-1) != module.input_dim:
            raise ValueError(
                "Feature dimensions do not match the tensor-product irreps."
            )
        features = indexed_sum(
            features,
            module.harmonic_input_index,
            module.harmonic_input_ptr,
            module.harmonic_input_transpose_index,
            module.harmonic_input_transpose_ptr,
            module.harmonic_input_count,
        )
    elif not module.input_identity:
        features = features.index_select(-1, module.input_index)
    if vectors is None and (not module.attrs_identity):
        edge_attrs = edge_attrs.index_select(-1, module.attrs_index)
    output = O3TensorProductConv.forward(
        module,
        features,
        edge_attrs,
        radial,
        projection,
        edge_index,
        num_nodes,
        vectors=vectors,
        amplitudes=amplitudes,
        radial_network=radial_network,
    )
    if vectors is not None and module.compact_output:
        return output
    if vectors is not None:
        output = output.index_select(-1, module.harmonic_output_index)
    elif not module.output_identity:
        output = output.index_select(-1, module.output_inverse)
    if module.compact_output:
        return output.index_select(-1, module.output_pack_index)
    return module.projector(output) if not config.TORCH_REFERENCE else output


def reference(module, features, edge_attrs, radial, projection, edge_index, num_nodes):
    """Evaluate each Cartesian path with ordinary delta/epsilon contractions."""
    from ....common.torch.cartesian import cartesian_product
    from ...co3.basis import project_reference

    features = (
        features
        if module.input_identity
        else features.index_select(-1, module.input_inverse)
    )
    edge_attrs = (
        edge_attrs
        if module.attrs_identity
        else edge_attrs.index_select(-1, module.attrs_inverse)
    )
    indexed = edge_index is not None
    source = features[edge_index[0]] if indexed else features
    weights = radial @ projection if projection.numel() else radial
    sections = [irreps.slices() for irreps in (module.irreps_in1, module.irreps_in2)]
    blocks, offset = [None] * len(module.irreps_out), 0
    for instruction, scale in zip(module.instructions, module.tp.path_factors):
        u, left = module.irreps_in1[instruction.i_in1]
        v, right = module.irreps_in2[instruction.i_in2]
        _, output = module.irreps_out[instruction.i_out]
        size = u * v if instruction.has_weight else 0
        matrix = weights[:, offset : offset + size] if size else None
        offset += size
        if not u or not v or not scale:
            continue
        x = source[:, sections[0][instruction.i_in1]].reshape(
            source.shape[0], u, left.dim
        )
        y = edge_attrs[:, sections[1][instruction.i_in2]].reshape(
            edge_attrs.shape[0], v, right.dim
        )
        value = cartesian_product(
            x.unsqueeze(-2), y.unsqueeze(-3), left.l, right.l, output.l
        )
        if matrix is not None:
            value = value * matrix.reshape(weights.shape[0], u, v, 1)
        value = value.sum(-2).flatten(1) * scale
        index = instruction.i_out
        blocks[index] = value if blocks[index] is None else blocks[index] + value
    values = [
        source.new_zeros((source.shape[0], mul, ir.dim))
        if block is None
        else block.reshape(source.shape[0], mul, ir.dim)
        for block, (mul, ir) in zip(blocks, module.irreps_out)
    ]
    if module.tp.project:
        values = project_reference(module.projector, values)
    message = (
        torch.cat([value.flatten(1) for value in values], -1)
        if values
        else source[:, :0]
    )
    output = (
        message.new_zeros((num_nodes, module.output_dim)).index_add(
            0, edge_index[1], message
        )
        if indexed
        else message
    )
    output = output + sum(
        (x.sum() * 0 for x in (features, edge_attrs, radial, projection))
    )
    return (
        output
        if module.output_identity
        else output.index_select(-1, module.output_index)
    )
