"""Edge-local irreducible Cartesian products and graph reduction."""

import math

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ...ops import config
from ...ops.common.radial import cast_radial, materialize, radial_features, radial_parts
from ...ops.common.torch.cartesian import coupling_size, normalized_coefficients
from ...ops.common.weights import IndexedProjection, prepare_indexed, radial_policy
from ...ops.torch.co3.ictp import tensor_product as torch_product
from ...utils.metadata import static_dimensions
from ...utils.program import Program
from .tensor_product import CartesianProduct, symmetric_entries

EDGE_TERMS = 4096
EDGE_WORKSPACE_BYTES = 8 << 20


def edge_coefficients(product, budget):
    """Return fully detraced edge coefficients, or use the factored evaluation."""
    ranks = product.l1, product.l2, product.l3
    if product.path_type == "direct":
        return (
            normalized_coefficients(*ranks) if coupling_size(*ranks) <= budget else None
        )
    if product.path_type == "symmetrize":
        return symmetric_entries(ranks, budget, False)
    if not getattr(product, "sparse_product", False):
        return None
    left, right, weights = product.sparse_coefficients[-3:]
    indices = product.indices.tolist()
    count = weights.count_nonzero(1).tolist()
    if sum(count[i] for i in indices) > budget:
        return None
    rows = [
        list(zip(a, b, w))
        for a, b, w in zip(left.tolist(), right.tolist(), weights.tolist())
    ]
    return tuple(
        (a, b, c, v) for c, i in enumerate(indices) for a, b, v in rows[i] if v
    )


def build(module):
    from ...conv.co3.polynomials import ANGULAR_TILE_SIZE, output_components

    tp = module.tp
    if tp.backend != module.backend:
        module.edge_products = torch.nn.ModuleDict(
            {
                key: CartesianProduct(p.l1, p.l2, p.l3, backend=module.backend)
                for key, p in tp.products.items()
            }
        )
    paths, used, offset = [], 0, 0
    sections = [ir.slices() for ir in (tp.irreps_in1, tp.irreps_in2, tp.irreps_out)]
    module.amplitude_dim = sum(mul for mul, _ in tp.irreps_in2)
    module.has_angles = any(mul and ir.l for mul, ir in tp.irreps_in2)
    module.harmonic_input_dim = module.input_dim
    module.analytic_literal = True
    for ins, scale in zip(tp.instructions, tp.path_factors):
        u, l = tp.irreps_in1[ins.i_in1]
        v, m = tp.irreps_in2[ins.i_in2]
        _, n = tp.irreps_out[ins.i_out]
        weight = offset if ins.has_weight else -1
        offset += math.prod(ins.path_shape) if ins.has_weight else 0
        if not u or not v or not scale:
            continue
        entries = edge_coefficients(
            tp.products[f"{l.l}_{m.l}_{n.l}"], EDGE_TERMS - used
        )
        if entries is None:
            module.analytic_literal = False
            return
        used += len(entries)
        paths.append(
            (
                sections[0][ins.i_in1].start,
                sections[1][ins.i_in2].start,
                sections[2][ins.i_out].start,
                u,
                v,
                l.dim,
                m.dim,
                n.dim,
                weight,
                scale,
                entries,
            )
        )
    unweighted = any(p[8] < 0 for p in paths)
    module.paths = tuple(paths)
    module.kernel_metadata = repr((module.paths, module.weight_numel, unweighted))
    paths, expansion, module.harmonic_output_dim = output_components(
        paths, tp.irreps_out, module.normalization
    )
    module.register_buffer(
        "harmonic_output_index",
        torch.tensor(expansion, dtype=torch.long)[module.output_inverse],
        persistent=False,
    )
    offsets, offset = {}, 0
    for (mul, _), section in zip(tp.irreps_in2, sections[1]):
        offsets[section.start] = offset
        offset += mul
    tiles = []
    for path in paths:
        groups = {}
        for a, b, c, v in path[-1]:
            tile = tuple(i // ANGULAR_TILE_SIZE for i in (a, b, c))
            groups.setdefault(tile, []).append((a, b, c, v))
        tiles.extend(
            (path[0], offsets[path[1]], *path[2:-1], tuple(entries))
            for entries in groups.values()
        )
    module.harmonic_metadata = repr(
        (
            tuple(tiles),
            module.weight_numel,
            unweighted,
            ("cartesian", module.normalization),
        )
    )


def attributes(module, vectors, amplitudes):
    if module.normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    values, offset = [], 0
    for mul, ir in module.irreps_in2:
        harmonic = module.harmonics[str(ir.l)](vectors).unsqueeze(-2)
        if amplitudes is None:
            value = harmonic.expand(-1, mul, -1)
        else:
            a = amplitudes.expand(amplitudes.shape[0], module.amplitude_dim)[
                :, offset : offset + mul
            ]
            value = harmonic * a.unsqueeze(-1)
        values.append(value.flatten(1))
        offset += mul
    result = (
        values[0]
        if len(values) == 1
        else torch.cat(values, -1)
        if values
        else vectors[:, :0]
    )
    return result if module.has_angles else result + vectors[:, :0].sum()


def edge_messages(module, x, y, weights, *, ordinary):
    """Apply the analytic path product before any graph reduction."""
    tp = module.tp
    products = getattr(module, "edge_products", tp.products)
    blocks = [None] * len(tp.irreps_out)
    left = x.split([mul * ir.dim for mul, ir in tp.irreps_in1], -1)
    right = y.split([mul * ir.dim for mul, ir in tp.irreps_in2], -1)
    offset = 0
    for ins, scale in zip(tp.instructions, tp.path_factors):
        u, l = tp.irreps_in1[ins.i_in1]
        v, m = tp.irreps_in2[ins.i_in2]
        _, n = tp.irreps_out[ins.i_out]
        w = weights[..., offset : offset + u * v] if ins.has_weight else None
        offset += u * v if ins.has_weight else 0
        if not u or not v or not scale:
            continue
        a, b = (
            left[ins.i_in1].unflatten(-1, (u, l.dim)),
            right[ins.i_in2].unflatten(-1, (v, m.dim)),
        )
        product = products[f"{l.l}_{m.l}_{n.l}"]
        if ordinary:
            z = torch_product.reference(product, a.unsqueeze(-2), b.unsqueeze(-3))
            if w is not None:
                z = z * w.unflatten(-1, (u, v)).unsqueeze(-1)
            z = z.sum(-2)
        elif v == 1:
            z = product(a, b)
            if w is not None:
                z = z * w.unsqueeze(-1)
        else:
            if w is not None:
                b = w.unflatten(-1, (u, v)) @ b
            else:
                b = b.sum(-2, keepdim=True)
            z = product(a, b)
        z = (z * scale).flatten(1)
        previous = blocks[ins.i_out]
        blocks[ins.i_out] = z if previous is None else previous + z
    rows = x.shape[0]
    out = [
        x.new_zeros(rows, mul * ir.dim) if z is None else z
        for z, (mul, ir) in zip(blocks, tp.irreps_out)
    ]
    result = out[0] if len(out) == 1 else torch.cat(out, -1) if out else x[:, :0]
    if all(z is None for z in blocks):
        result = result + x[..., :0].sum() + y[..., :0].sum() + weights[..., :0].sum()
    elif not tp.weight_numel:
        result = result + weights[..., :0].sum()
    return result


def radial_weights(network, radial, index, projection, backend):
    """Evaluate and recompute the radial MLP and its last linear within one tile."""
    from ...ops.common.convolution import radial_input
    from ...ops.common.edge import edge_program

    inputs, program = [], Program()
    value = radial_input(program, inputs, radial, network)
    width, columns = static_dimensions(projection)
    if width:
        if program.size(value) + 1 == width:
            value = program.concatenate((value, program.constant(1, 1)))
        if program.size(value) != width:
            raise ValueError("Radial network output does not match projection input.")
        matrix = program.input(len(inputs), "shared", width * columns)
        inputs.append(projection.reshape(1, -1))
        value = program.matmul(value, matrix, 1, width, columns)
    metadata = repr((tuple(program.nodes), ((value, 0, "edge"),)))
    return edge_program(backend, metadata, 0, index[0], index[1], *inputs)[0]


def evaluate(
    module, x, attrs, radial, projection, index, nodes, vectors, amplitudes, network
):
    from ...o3.cgtp.convolution import O3TensorProductConv

    if module.symmetric_inputs:
        reduced = x.new_zeros(x.shape[0], module.symmetric_dim).index_add(
            -1, module.symmetric_index, x
        )
        x = (reduced / module.symmetric_count).index_select(-1, module.symmetric_index)
    if x.ndim != 2 or x.shape[-1] != module.input_dim:
        raise ValueError("Node features must match the Cartesian irreps.")
    if index is not None and (index.ndim != 2 or index.size(0) != 2):
        raise ValueError("edge_index must have shape (2, edges).")
    if index is None and (
        vectors is not None or network is not None or projection.numel()
    ):
        raise ValueError("Row-wise products require explicit attributes and weights.")
    if vectors is not None:
        if vectors.ndim != 2 or vectors.shape != (index.size(1), 3):
            raise ValueError("vectors must have shape (edges, 3).")
        if amplitudes is not None and (
            amplitudes.ndim != 2
            or amplitudes.shape[0] not in (1, index.size(1))
            or amplitudes.shape[1] not in (1, module.amplitude_dim)
        ):
            raise ValueError(
                "Amplitudes must broadcast over edges and harmonic channels."
            )
    elif attrs is None:
        raise ValueError("Provide edge attributes or vectors.")
    native = module.backend == "cuda" and x.is_cuda
    ordinary = not native and config.TORCH_REFERENCE
    if (
        module.analytic_literal
        and not ordinary
        and (
            native or (network is not None and radial_policy(module, network) == "full")
        )
    ):
        if vectors is not None and not (native and vectors.dtype == x.dtype):
            attrs = attributes(module, vectors, amplitudes)
            vectors = None
        features = (
            x if module.input_identity else x.index_select(-1, module.input_index)
        )
        if vectors is None and not module.attrs_identity:
            attrs = attrs.index_select(-1, module.attrs_index)
        out = O3TensorProductConv.forward(
            module,
            features,
            attrs,
            radial,
            projection,
            index,
            nodes,
            vectors=vectors,
            amplitudes=amplitudes,
            radial_network=network,
        )
        if vectors is not None:
            return out.index_select(-1, module.harmonic_output_index)
        return (
            out
            if module.output_identity
            else out.index_select(-1, module.output_inverse)
        )

    radial = cast_radial(radial, x.dtype)
    projection = projection.to(x)
    if isinstance(network, IndexedProjection):
        radial, network = prepare_indexed(module, radial, network, index, projection)
    rows = x.size(0) if index is None else index.size(1)
    nodes = x.size(0) if nodes is None else nodes
    if index is None and nodes != rows:
        raise ValueError("Row-wise products require one output row per input row.")
    if projection.ndim != 2 or projection.size(1) != module.weight_numel:
        raise ValueError(
            "Edge attributes or radial projection do not match the convolution."
        )
    if vectors is None:
        if (
            attrs.ndim != 2
            or attrs.size(-1) != module.edge_dim
            or attrs.size(0) not in (1, rows)
        ):
            raise ValueError("Edge attributes do not match the convolution.")
        attrs = attrs.to(x)
    parts = radial_parts(radial)
    if index is None and any(kind not in ("edge", "shared") for _, kind in parts):
        raise ValueError("Indexed radial features require edge_index.")
    if any(kind == "edge" and val.size(0) != rows for val, kind in parts):
        raise ValueError(
            "Radial features must have one row per edge or one shared row."
        )
    if network is None and sum(val.size(1) for val, _ in parts) != (
        projection.size(0) or module.weight_numel
    ):
        raise ValueError(
            "Radial input width does not match the projection or path weights."
        )
    recompute = (
        network is not None
        and radial_policy(module, network) == "full"
        and not ordinary
    )
    tile = max(
        1,
        EDGE_WORKSPACE_BYTES
        // (
            x.element_size()
            * max(
                1,
                module.input_dim
                + module.edge_dim
                + module.output_dim
                + module.weight_numel,
            )
        ),
    )
    out = x.new_zeros(nodes, module.output_dim)
    per_edge = []
    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        # A dynamic edge count must not specialize a Python loop during export.
        tiles = ((0, rows),)
    else:
        tiles = (
            (start, min(start + tile, rows)) for start in range(0, max(1, rows), tile)
        )
    for start, stop in tiles:
        e = None if index is None else index[:, start:stop]
        r = tuple((v[start:stop] if kind == "edge" else v, kind) for v, kind in parts)
        local_network = (
            network.narrow_edges(start, stop)
            if isinstance(network, IndexedProjection)
            else network
        )
        if network is None:
            if len(r) == 1 and r[0][1] in ("edge", "shared"):
                w = r[0][0]
            elif e is not None:
                w = materialize(r, e)
            else:
                w = torch.cat([v.expand(stop - start, -1) for v, _ in r], -1)
        elif recompute:
            w = radial_weights(
                local_network, r, e, projection, "cuda" if native else "torch"
            )
        else:
            w = radial_features(local_network, r, e, projection, reference=ordinary)
        if projection.numel() and not recompute:
            w = w @ projection
        if vectors is None:
            y = attrs if attrs.size(0) == 1 else attrs[start:stop]
        else:
            amplitude = (
                amplitudes
                if amplitudes is None or amplitudes.size(0) == 1
                else amplitudes[start:stop]
            )
            y = attributes(module, vectors[start:stop], amplitude).to(x)
        z = edge_messages(
            module,
            x[start:stop] if e is None else x.index_select(0, e[0]),
            y,
            w,
            ordinary=ordinary,
        )
        if e is None:
            per_edge.append(z)
        else:
            out.index_add_(0, e[1], z)
    return torch.cat(per_edge, 0) if index is None else out
