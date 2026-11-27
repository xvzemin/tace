"""Parameter sharing between message-passing layers."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import gcd, prod

import torch
from torch import nn
from torch.nn.utils import parametrize

from equivariantx.nn import EquivariantLayerNorm

from ..linear import e3nnElementLinear, e3nnLinear, e3nnMoEElementLinear
from .edge import Element2EdgeUpdate, ElementEdgeEmbedding, IdentityEdgeUpdate
from .inter import O2CgtpInteraction, O3CgtpInteraction, O3IctdInteraction
from .node import Element2NodeUpdate, ElementNodeUpdate
from .paths import symmetric_contraction_paths
from .scatter_norm import DensityScatterNorm


def label(ir):
    return ir.l, ir.p, getattr(ir, "t", 1)


SCALAR = (0, 1, 1)


def fields(irreps):
    copies = Counter()
    result = []
    for mul, ir in irreps:
        key = label(ir)
        result.append((key, (key, copies[key]), mul))
        copies[key] += 1
    return result


def coupled(left, right, instructions):
    copies = Counter()
    result = []
    for i, j, ir in instructions:
        key = (left[i][1], right[j][1], label(ir))
        result.append((label(ir), (key, copies[key]), left[i][2]))
        copies[key] += 1
    return result


def product_fields(tp, left, right):
    paths = sorted(tp.instructions, key=lambda ins: ins[2])
    if [ins[2] for ins in paths] != list(range(len(tp.irreps_out))):
        raise ValueError("loop requires a distinct output block per CGTP path.")
    return coupled(left, right, [(i, j, tp.irreps_out[k].ir) for i, j, k, *_ in paths])


def partition(irreps, blocks, experts):
    position = 0
    result = []
    for mul, ir in irreps:
        rows = []
        used = 0
        while used < mul and position < len(blocks):
            kind, key, width = blocks[position]
            if kind != label(ir) or width % experts:
                raise ValueError("loop: incompatible irrep or expert channel layout.")
            rows.append((key, width // experts))
            used += width
            position += 1
        if used != mul:
            raise ValueError("loop: incompatible multiplicity layout.")
        result.append(rows)
    if position != len(blocks):
        raise ValueError("loop: unexpected channels after the final irrep.")
    return result


def sorted_fields(blocks):
    # e3nn's sort is stable for equal irreps.
    return sorted(blocks, key=lambda b: b[0])


def matrix_blocks(inputs, outputs):
    return tuple(
        ((a, row, b), width)
        for a, rows in inputs
        for row in range(rows)
        for b, width in outputs
    )


def local_linear_axes(conv, axes):
    frames = [ElementEdgeEmbedding.frame_channels(conv.local_frame_in)] * 2
    if hasattr(conv, "magnetic_frame"):
        frames.append(ElementEdgeEmbedding.frame_channels(conv.magnetic_frame))
    inputs = [
        [
            ((ir, side, entry), conv.num_channel)
            for side, frame in enumerate(frames)
            for entry in frame.get(ir, ())
        ]
        for ir, _ in conv.local_irreps_in
    ]
    outputs = ElementEdgeEmbedding.frame_channels(conv.local_frame_out)
    gate = conv.nonlinearity

    def copies(ir, role):
        return [((role, ir, entry), conv.num_channel) for entry in outputs[ir]]

    gate_in = [[] for _ in gate.irreps_in]
    gate_out = []
    for location, (ir, _) in zip(gate._scalar_locations, gate.irreps_scalars):
        entries = copies(ir, "scalar")
        gate_in[location].extend(entries)
        gate_out.append(entries)
    for p in gate._paths:
        role = (p.ir_gate, p.ir_gated)
        entries = copies(p.ir_out, role)
        if sum(width for _, width in entries) != p.mul:
            raise ValueError("loop: unsupported split local gate channels.")
        gate_in[gate._gate_locations[p.i_gate]].extend(
            [(("gate", k), n) for k, n in entries]
        )
        gate_in[gate._gated_locations[p.i_gated]].extend(entries)
        gate_out.append(entries)
    for name, source, target in (
        ("linear_up", inputs, gate_in),
        (
            "linear_down",
            gate_out,
            [copies(ir, "out") for ir, _ in conv.local_irreps_out],
        ),
    ):
        linear = getattr(conv, name)
        weights, bias = [], []
        for ins in linear.instructions:
            if ins.i_in == -1:
                bias.extend(target[ins.i_out])
            else:
                weights.extend(matrix_blocks(source[ins.i_in], target[ins.i_out]))
        axes[f"rejector.{name}.weight"] = (-1, tuple(weights))
        axes[f"rejector.{name}.bias"] = (-1, tuple(bias))


def layouts(layer):
    """Return channel identities before simplification of CGTP and ACE irreps."""
    inputs, outputs, axes = {}, {}, {}
    if hasattr(layer, "rejector"):
        conv = layer.rejector
        if hasattr(conv, "local_frame_in") and hasattr(
            getattr(conv, "nonlinearity", None), "_paths"
        ):
            local_linear_axes(conv, axes)
        tp = conv if hasattr(conv, "instructions") else getattr(conv, "tp", None)
        if hasattr(conv, "magnetic_weight_numel"):
            first, second = conv.eqx_tp.tp1, conv.eqx_tp.tp2
            middle = product_fields(
                first, fields(first.irreps_in1), fields(first.irreps_in2)
            )
            final = product_fields(second, middle, fields(second.irreps_in2))
            inputs["linear_down"] = final
            for prefix, net, stage, rows in (
                ("edge_info", layer.edge_info, first, middle),
                ("magnetic_edge_info", layer.magnetic_edge_info, second, final),
            ):
                items = tuple(
                    (rows[ins.i_out][1], prod(ins.path_shape))
                    for ins in stage.instructions
                    if ins.has_weight
                )
                key = f"{prefix}.mlp.{len(net.mlp) - 1}"
                axes[key + ".weight"] = axes[key + ".bias"] = (-1, items)
        if hasattr(conv, "irreps_in2") and hasattr(tp, "instructions"):
            inputs["linear_down"] = product_fields(
                tp, fields(conv.irreps_in1), fields(conv.irreps_in2)
            )
            conv = tp
        if (hasattr(conv, "irreps_in2") and hasattr(conv, "instructions")) or hasattr(
            conv, "local_frame_in"
        ):
            keys = ElementEdgeEmbedding.path_keys(conv)
            network = layer.edge_info
            if hasattr(network, "branches"):
                groups = [
                    (
                        f"edge_info.branches.{i}",
                        branch,
                        [k for k in keys if k[2][0] == degree],
                    )
                    for i, (degree, branch) in enumerate(
                        zip(network.active, network.branches)
                    )
                ]
            else:
                groups = [("edge_info", network, keys)]
            for name, net, cols in groups:
                name = f"{name}.mlp.{len(net.mlp) - 1}"
                items = tuple((k, 1) for k in cols)
                axes[name + ".weight"] = (-1, items)
                axes[name + ".bias"] = (-1, items)
        if layer.nonlinear is not None:
            features = [
                (ir, key, layer.num_channel) for ir, key, _ in fields(layer.irreps_out)
            ]
            scalar = [] if layer.gate_m0 else [b for b in features if b[0][0] == 0]
            gated = [b for b in features if b not in scalar]
            gates = [(SCALAR, ("gate", key), n) for _, key, n in gated]
            outputs["linear_down"] = sorted_fields(scalar + gates + gated)
    elif hasattr(layer, "aces"):
        base = fields(layer.irreps_hidden)
        right = fields(getattr(layer, "irreps_base", layer.irreps_hidden))
        orders = [base]
        if hasattr(layer, "fully_connected"):
            paths = layer.instructions
            if paths is None:
                paths = symmetric_contraction_paths(
                    layer.irreps_hidden, layer.irreps_tp_out_list, l1l2=layer.l1l2
                )
            for stage in paths:
                orders.append(coupled(orders[-1], right, stage))
        else:
            for tp in layer.aces:
                orders.append(product_fields(tp, orders[-1], right))
        for i, order in enumerate(orders):
            inputs[f"coefs.{i}"] = inputs[f"shared_coefs.{i}"] = order
        if getattr(layer, "use_bilinear_gate", False):
            gates = [(SCALAR, ("gate", k), n) for _, k, n in orders[1]]
            factors = [(ir, ("factor", i, k), n) for ir, k, n in base for i in (0, 1)]
            outputs["linear_up"] = gates + factors
        elif hasattr(layer, "num_inputs"):
            scalar = [(SCALAR, "summary", layer.num_hidden_channel)]
            factors = [
                (ir, ("factor", i, k), n)
                for i in range(layer.num_inputs)
                for ir, k, n in base
            ]
            outputs["linear_up"] = sorted_fields(scalar + factors)
            columns = tuple(
                ((i, k), n) for i, order in enumerate(orders) for _, k, n in order
            )
            axes["gate.0.weight"] = axes["gate.0.bias"] = (0, columns)
        elif hasattr(layer, "fully_connected"):
            width = layer.num_hidden_channel**2 if layer.fully_connected else 1
            kept = {label(ir) for _, ir in layer.irreps_coefs_out}
            gates = [
                (SCALAR, ("gate", i, k), width)
                for i, order in enumerate(orders)
                for ir, k, _ in order
                if ir in kept
            ]
            factors = [
                (ir, ("factor", i, k), n)
                for i in range(layer.correlation)
                for ir, k, n in base
            ]
            outputs["linear_up"] = sorted_fields(gates + factors)
    elif hasattr(layer, "coupling_blocks"):
        axes["weight"] = (
            1,
            tuple(
                ((label(ir), degree, tuple(fields(layer.irreps_hidden))), n)
                for ir, degree, n in layer.coupling_blocks
            ),
        )
    return inputs, outputs, axes


@dataclass
class Slot:
    module: nn.Module
    name: str
    axis: int = 0
    ndim: int = 1
    blocks: tuple | None = None
    signature: tuple = ()

    @property
    def value(self):
        return self.module.get_parameter(self.name)


def expert_blocks(entries, expert, experts):
    if experts == 1:
        return entries
    size = sum(width for _, width in entries)
    start, stop = expert * size, (expert + 1) * size
    result, offset = [], 0
    for key, width in entries:
        for tile in range(experts):
            a, b = max(start, offset), min(stop, offset + width)
            if a < b:
                if a != offset or b != offset + width:
                    raise ValueError(
                        "loop: expert partitions split a path's channel tile."
                    )
                result.append(((expert, key, tile), width))
            offset += width
    return result


def slots(layer):
    inputs, outputs, axes = layouts(layer)
    result = {}
    for name, module in layer.named_modules():
        if not isinstance(
            module, (e3nnLinear, e3nnElementLinear, e3nnMoEElementLinear)
        ):
            continue
        experts = getattr(module, "num_experts", 1)
        ins = partition(
            module.irreps_in, inputs.get(name, fields(module.irreps_in)), experts
        )
        outs = partition(
            module.irreps_out, outputs.get(name, fields(module.irreps_out)), experts
        )
        flat = [[] for _ in range(experts)]
        extra = int(experts > 1)
        for idx, instruction in enumerate(module.linear.instructions):
            i, j = instruction.i_in, instruction.i_out
            by_expert = [
                matrix_blocks(
                    expert_blocks(ins[i], e, experts),
                    expert_blocks(outs[j], e, experts),
                )
                for e in range(experts)
            ]
            if isinstance(module.weight, nn.ParameterList):
                identity = (
                    fields(module.irreps_in)[i][1],
                    fields(module.irreps_out)[j][1],
                )
                result[name, "matrix", identity] = Slot(
                    module.weight,
                    str(idx),
                    -2 - extra,
                    2 + extra,
                    tuple(b for rows in by_expert for b in rows),
                )
            else:
                for dest, rows in zip(flat, by_expert):
                    dest.extend(rows)
        if isinstance(module.weight, nn.Parameter):
            result[name, "weight"] = Slot(
                module,
                "weight",
                -1 - extra,
                1 + extra,
                tuple(b for rows in flat for b in rows),
            )
        if module.bias is not None:
            blocks = tuple(
                b
                for e in range(experts)
                for (_, ir), entries in zip(module.irreps_out, outs)
                if ir.is_scalar()
                for b in expert_blocks(entries, e, experts)
            )
            if module.use_matrix_weight and isinstance(module, e3nnElementLinear):
                blocks = tuple(
                    ((e, k), n) for e in range(module.num_elements) for k, n in blocks
                )
            result[name, "bias"] = Slot(module, "bias", -1 - extra, 1 + extra, blocks)

    covered = {id(slot.value) for slot in result.values()}
    for path, parameter in layer.named_parameters():
        if id(parameter) in covered:
            continue
        name, _, attr = path.rpartition(".")
        module = layer.get_submodule(name)
        slot = Slot(module, attr)
        if path in axes:
            slot.axis, slot.blocks = axes[path]
        elif isinstance(module, EquivariantLayerNorm):
            slot.blocks = tuple(
                (k, n)
                for ir, k, n in fields(module.irreps)
                if attr == "weight" or ir == SCALAR
            )
        # Opaque local operators have no general coordinate map between different frames.
        if (
            path not in axes
            and path.startswith("rejector.")
            and hasattr(layer.rejector, "local_frame_in")
        ):
            slot.signature = (str(layer.irreps_in), str(layer.irreps_out))
        result[name, attr] = slot
    network = getattr(layer, "edge_info", None)
    if hasattr(network, "branches"):
        for key in list(result):
            parts = key[0].split(".")
            if parts[:2] == ["edge_info", "branches"]:
                parts[2] = str(network.active[int(parts[2])])
                parts[1] = "degree"
                result[(".".join(parts), *key[1:])] = result.pop(key)
    return result


class SelectPaths(nn.Module):
    """Select a layer's ordered path weights from a shared parameter."""

    def __init__(self, axis, ndim, bank_size, target_shape, spans):
        super().__init__()
        self.axis, self.ndim, self.shape = axis, ndim, tuple(target_shape)
        runs = []
        for span in spans:
            if runs and runs[-1].stop == span.start:
                runs[-1] = range(runs[-1].start, span.stop)
            else:
                runs.append(span)
        self.start = runs[0].start if runs else 0
        self.length = sum(map(len, runs))
        self.block = gcd(bank_size, *(v for r in runs for v in (r.start, len(r)))) or 1
        index = None
        if len(runs) > 1:
            index = torch.tensor(
                [
                    i
                    for r in runs
                    for i in range(r.start // self.block, r.stop // self.block)
                ],
                dtype=torch.long,
            )
        self.register_buffer("index", index, persistent=False)

    def forward(self, weight):
        weight = weight.flatten(self.axis, self.axis + self.ndim - 1)
        if self.index is None:
            weight = weight.narrow(self.axis, self.start, self.length)
        elif self.block == 1:
            weight = weight.index_select(self.axis, self.index)
        else:
            weight = weight.unflatten(self.axis, (-1, self.block)).index_select(
                self.axis, self.index
            )
        return weight.reshape(self.shape)


def tie(slots, context):
    if len(slots) < 2:
        return
    source = max(slots, key=lambda s: s.value.numel())
    if any(s.signature != source.signature for s in slots):
        raise ValueError(f"loop: no common path layout at {context}.")
    if all(
        s.blocks == source.blocks and s.value.shape == source.value.shape for s in slots
    ):
        for s in slots:
            setattr(s.module, s.name, source.value)
        return
    if any(s.blocks is None for s in slots):
        raise ValueError(f"loop: incompatible parameter layouts at {context}.")

    def spans(s):
        offset, result = 0, {}
        for key, width in s.blocks:
            if key in result:
                raise ValueError(f"loop: ambiguous path at {context}.")
            result[key] = range(offset, offset + width)
            offset += width
        axis = s.axis % s.value.ndim
        if offset != prod(s.value.shape[axis : axis + s.ndim]):
            raise ValueError(f"loop: path layout does not match {context}.")
        return result

    maps = [spans(s) for s in slots]
    bank = spans(source)
    axis = source.axis % source.value.ndim
    shape = source.value.flatten(axis, axis + source.ndim - 1).shape
    union = dict(bank)
    size = shape[axis]
    for s, layout in zip(slots, maps):
        dim = s.axis % s.value.ndim
        dims = s.value.flatten(dim, dim + s.ndim - 1).shape
        if (
            dim != axis
            or dims[:dim] + dims[dim + 1 :] != shape[:axis] + shape[axis + 1 :]
        ):
            raise ValueError(f"loop: incompatible independent dimensions at {context}.")
        for key, span in layout.items():
            if key not in union:
                union[key] = range(size, size + len(span))
                size += len(span)
            elif len(union[key]) != len(span):
                raise ValueError(f"loop: incompatible channel widths at {context}.")
    master = source.value
    master_ndim = source.ndim
    if size != shape[axis]:
        values = master.new_empty((*shape[:axis], size, *shape[axis + 1 :]))
        with torch.no_grad():
            for s, layout in zip(slots, maps):
                flat = s.value.flatten(axis, axis + s.ndim - 1)
                for key, span in layout.items():
                    values.narrow(axis, union[key].start, len(span)).copy_(
                        flat.narrow(axis, span.start, len(span))
                    )
        master = nn.Parameter(values, requires_grad=master.requires_grad)
        master_ndim = 1
    for s, layout in zip(slots, maps):
        if (
            size == shape[axis]
            and s.blocks == source.blocks
            and s.value.shape == master.shape
        ):
            setattr(s.module, s.name, master)
        else:
            view = SelectPaths(
                axis, master_ndim, size, s.value.shape, (union[key] for key in layout)
            )
            parametrize.register_parametrization(s.module, s.name, view, unsafe=True)
            s.module.parametrizations[s.name].original = master


def grouped_layers(loop):
    groups = defaultdict(list)
    for i, group in enumerate(loop):
        groups[group].append(i)
    return groups


def layer_groups(representation, interaction_loop, product_loop):
    """Describe the shared parameters in the model's native path order."""
    groups = grouped_layers(interaction_loop)
    sources = []
    for i, g in enumerate(interaction_loop):
        first = groups[g][0] if len(groups[g]) > 1 else -1
        inter = representation.interactions[i]
        update = type(representation.edge_updates[i])
        static_edge = update in (Element2EdgeUpdate, IdentityEdgeUpdate)
        static_node = representation.node_updates is not None and type(
            representation.node_updates[i]
        ) in (ElementNodeUpdate, Element2NodeUpdate)
        static_radial = (
            static_edge
            and type(inter) in (O3CgtpInteraction, O3IctdInteraction, O2CgtpInteraction)
            and inter.radial_widths is None
            and inter.edge_info.num_layers > 1
        )
        sources.append(
            (
                first if update is Element2EdgeUpdate else -1,
                first if static_node else -1,
                first
                if static_edge and isinstance(inter.scatter_norm, DensityScatterNorm)
                else -1,
                first if static_radial else -1,
            )
        )
    representation.loop_update_sources = tuple(sources)
    products = grouped_layers(product_loop)
    for name, schedule in (
        ("interactions", groups),
        ("edge_updates", groups),
        ("node_updates", groups),
        ("magnetic_edge_updates", groups),
        ("uee_embeddings", products),
        ("products", products),
    ):
        layers = getattr(representation, name)
        if layers is None:
            continue
        for group, indices in schedule.items():
            if len(indices) == 1:
                continue
            members = [layers[i] for i in indices]
            if len({type(layer) for layer in members}) != 1:
                raise ValueError(f"loop group {group} mixes {name} types.")
            for attr in ("nonlinear", "edge_nonlinear"):
                if len({repr(getattr(layer, attr, None)) for layer in members}) != 1:
                    raise ValueError(
                        f"loop group {group} has incompatible {name}.{attr}."
                    )
            entries = defaultdict(list)
            for idx, layer in zip(indices, members):
                names = {id(module): path for path, module in layer.named_modules()}
                for key, slot in slots(layer).items():
                    path = names[id(slot.module)]
                    entries[key].append(
                        (f"{name}.{idx}" + (f".{path}" if path else ""), slot)
                    )
            for key, entries_for_key in entries.items():
                yield (group, name, key), entries_for_key


def share_layers(representation, groups):
    """Bind native operators to the shared parameters of each loop group."""
    for context, members in groups:
        for path, slot in members:
            slot.module = representation.get_submodule(path)
        tie([s for _, s in members], context)
    aliases = defaultdict(list)
    for name, param in representation.named_parameters(remove_duplicate=False):
        aliases[id(param)].append(name)
    representation.loop_aliases = tuple(
        tuple(names) for names in aliases.values() if len(names) > 1
    )
    for layer in representation.modules():
        if hasattr(layer, "use_matrix_weight"):
            layer.loop_weight_layout = True
    representation.register_load_state_dict_post_hook(restore_after_load)


def restore_aliases(representation):
    for canonical, *others in representation.loop_aliases:
        parameter = representation.get_parameter(canonical)
        for path in others:
            parent, _, attr = path.rpartition(".")
            setattr(representation.get_submodule(parent), attr, parameter)


def restore_after_load(module, incompatible_keys):
    restore_aliases(module)
