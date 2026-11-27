"""One-time binding of a joint AOTI training package to its TACE host."""

import hashlib

import torch

from equivariantx_foundation.registry import REGISTRATION_MODULES

# Only these reviewed, side-effect-free ATen overloads may be
# rescheduled. Unknown ops, custom kernels, random draws, assertions, writes
# and unused nodes retain their order and are fences for all other nodes.
_MOVABLE = frozenset(
    getattr(getattr(torch.ops.aten, name), overload)
    for name, overload in (
        item.split(".")
        for item in """
        _to_copy.default _unsafe_view.default add.Tensor alias.default arange.default as_strided.default bmm.default
        div.Tensor expand.default index.Tensor index_add.default index_put.default
        mm.default mul.Tensor neg.default new_zeros.default permute.default rsub.Scalar
        scalar_tensor.default select.int select_backward.default sigmoid.default
        sigmoid_backward.default slice.Tensor slice_backward.default squeeze.dim sum.default sum.dim_IntList t.default
        transpose.int unsqueeze.default view.default where.self zeros_like.default
        """.split()
    )
)


def _canonical_nodes(graph):
    """Visit ordered operands from ordered effects/outputs, retaining sharing.

    Independent autograd branches can be scheduled differently on a second
    trace. Each node carries its original fence interval; traversal may not
    silently move a read across a write to an alias, RNG or opaque kernel.
    """
    nodes = list(graph.nodes)
    positions = {node: i for i, node in enumerate(nodes)}
    epochs, roots, epoch = {}, [], 0
    for node in nodes:
        epochs[node] = epoch
        if node.op == "placeholder":
            roots.append(node)
            continue
        if node.op == "call_function" and node.target in _MOVABLE and node.users:
            continue
        if (
            node.target is torch.ops.aten.new_empty.default
            and node.users
            and all(
                user.target is torch.ops.aten.zeros_like.default for user in node.users
            )
        ):
            # The fresh uninitialized contents never escape: zeros_like reads
            # layout/options only. Every other new_empty remains a full fence.
            continue
        roots.append(node)
        if node.target is torch.ops.aten.copy_.default:
            # A copy into a single, previously unobserved view of a FRESH
            # zero allocation cannot affect independent reads. Still order
            # the copy w.r.t. other effects. Any other prior reader/escape
            # makes this a full fence, as does every other mutation.
            dst = node.args[0]
            if dst.target is torch.ops.aten.as_strided.default:
                alloc = dst.args[0]
                if (
                    alloc.target is torch.ops.aten.new_zeros.default
                    and all(
                        u is dst or positions[u] > positions[node] for u in alloc.users
                    )
                    and all(positions[u] >= positions[node] for u in dst.users)
                ):
                    continue
        epoch += 1

    seen = set()
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if node in seen:
                continue
            if done:
                seen.add(node)
                yield node, epochs[node]
            else:
                stack.append((node, True))
                stack.extend((n, False) for n in reversed(node.all_input_nodes))
    if len(seen) != len(nodes):
        raise ValueError("unreachable node in training trace")


def _graph_contract(trace):
    from tace.models.compile.artifact import json_bytes

    device = trace.args[0].device

    def literal(value):
        if isinstance(value, torch.fx.Node):
            return refs[value]
        if isinstance(value, torch.device):
            if value != device and value.type != "cpu":
                raise ValueError("trace uses a second accelerator")
            return {"device": value.type}
        if isinstance(value, (torch.dtype, torch.layout, torch.memory_format)):
            return str(value)
        if isinstance(value, (tuple, list)):
            return [literal(v) for v in value]
        if isinstance(value, dict):
            return {k: literal(v) for k, v in value.items()}
        if isinstance(value, slice):
            return {"slice": literal((value.start, value.stop, value.step))}
        if value is None or type(value) in (bool, str, int, float):
            return value
        raise ValueError(f"unsupported trace literal: {type(value).__name__}")

    def tensor_digest(tensor):
        if tensor.is_meta or tensor.layout != torch.strided:
            raise ValueError("unsupported trace tensor")
        raw = tensor.detach().resolve_conj().resolve_neg().contiguous().cpu()
        return (
            list(tensor.shape),
            list(tensor.stride()),
            str(tensor.dtype),
            tensor.device.type,
            hashlib.sha256(raw.reshape(-1).view(torch.uint8).numpy()).hexdigest(),
        )

    live_storage = {v.untyped_storage()._cdata: v.untyped_storage() for v in trace.args}
    # from_numpy/frombuffer can expose the same memory through *different*
    # PyTorch storage handles. Never certify independent ABI slots in that case.
    ranges = sorted(
        (str(s.device), s.data_ptr(), s.data_ptr() + s.nbytes())
        for s in live_storage.values()
        if s.nbytes()
    )
    if any(d == e and x < end for (d, _, end), (e, x, _) in zip(ranges, ranges[1:])):
        raise ValueError("overlapping input allocations have distinct storage handles")
    input_names = (*trace.batch_keys, *trace.parameters, *trace.buffers)
    if len(set(input_names)) != len(input_names):
        raise ValueError("ambiguous batch/state names")
    names = iter(input_names)
    refs, constants, namespaces = {}, [], set()
    topology = hashlib.sha256()
    count = 0
    for node, epoch in _canonical_nodes(trace.graph.graph):
        if node.op == "placeholder":
            refs[node] = ("input", next(names))
            continue
        refs[node] = ("node", count)
        count += 1
        if node.op == "get_attr":
            value = trace.graph
            for key in node.target.split("."):
                value = getattr(value, key)
            if not isinstance(value, torch.Tensor) or value.requires_grad:
                raise ValueError("unsupported or trainable captured constant")
            storage = value.untyped_storage()
            addr, end = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
            if storage._cdata in live_storage or (
                end > addr
                and any(
                    d == str(storage.device) and x < end and addr < y
                    for d, x, y in ranges
                )
            ):
                raise ValueError("a captured constant aliases live host state")
            constants.append(tensor_digest(value))
            row = ("constant", len(constants) - 1)
        elif node.op == "call_function":
            op = node.target
            if isinstance(op, torch._ops.OpOverload):
                namespaces.add(op.namespace)
                target = str(op)
            elif getattr(op, "__module__", None) in (
                "operator",
                "_operator",
                "math",
                "builtins",
            ):
                target = f"{op.__module__}.{op.__qualname__}"
            else:
                raise ValueError(f"unsupported training graph callable {op}")
            # Messages include tracer-generated SymInt/node names, not semantics.
            args = (
                node.args[:1]
                if op == torch.ops.aten._assert_scalar.default
                else node.args
            )
            row = (target, literal(args), literal(node.kwargs))
        elif node.op == "output":
            row = ("output", literal(node.args))
        else:
            raise ValueError(f"unsupported training graph node: {node.op}")
        topology.update(json_bytes((epoch, row)))
        topology.update(b"\n")
    custom = namespaces.difference({"aten", "prims"})
    if custom.difference({"eqx"}, REGISTRATION_MODULES):
        raise ValueError(f"unsupported custom operator namespaces: {sorted(custom)}")

    return topology.hexdigest(), constants, custom


def _training_host_contract(model, objective, trace):
    """Fingerprint the explicit forward/gradient program, not its live weights."""
    from tace.models.compile.artifact import (
        json_bytes,
        module_context,
        source_fingerprint,
    )

    device = trace.args[0].device
    state = torch.nn.ModuleDict({"model": model, "loss_fn": objective})
    if (
        tuple(dict(state.named_parameters())) != trace.parameters
        or tuple(dict(state.named_buffers())) != trace.buffers
    ):
        raise ValueError(
            "compiled adapter does not represent all registered host state"
        )
    storage_groups, objects, layout = {}, {}, []
    for kind, entries in (
        ("parameter", state.named_parameters(remove_duplicate=False)),
        ("buffer", state.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in entries:
            storage = tensor.untyped_storage()
            group = storage_groups.setdefault(storage._cdata, len(storage_groups))
            layout.append(
                (
                    kind,
                    name,
                    objects.setdefault(id(tensor), name),
                    group,
                    list(tensor.shape),
                    list(tensor.stride()),
                    tensor.storage_offset(),
                    str(tensor.dtype),
                    tensor.device.type,
                    tensor.requires_grad,
                )
            )
    topology, constants, custom = _graph_contract(trace)
    readout = model.readout_fn
    cutoff = getattr(readout, "cutoff", None)
    pair_cutoff = getattr(getattr(readout, "pair_representation", None), "cutoff", None)
    physical = (
        sorted(model.get_target_property()),
        sorted(model.get_embedding_property()),
        model.num_fidelities,
        model.uses_reciprocal_weights(),
        None if cutoff is None else float(cutoff),
        None if pair_cutoff is None else float(pair_cutoff),
        getattr(readout, "max_neighbors", None),
        model.get_atomic_numbers(),
    )
    precision = (
        str(torch.__version__),
        str(torch.get_default_dtype()),
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        None
        if device.type != "cuda"
        else (
            torch.cuda.get_device_capability(device),
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        ),
    )

    def digest(value):
        return hashlib.sha256(json_bytes(value)).hexdigest()

    # An EQX op may have been decomposed into native kernels at export, even
    # though its name remains in this frontend trace. Match the installed op
    # sources, not unrelated serving code, benchmarks or documentation.
    packages = ("equivariantx",) if custom else ()
    if custom.intersection(REGISTRATION_MODULES):
        packages += ("equivariantx_foundation.ops", "equivariantx_foundation.models")
    return {
        "graph": topology,
        "constants": digest(constants),
        "state": digest((layout, trace.parameters, trace.buffers, trace.gradients)),
        "predictions": digest(trace.predictions),
        "objective": module_context(objective),
        "geometry": digest(physical),
        "precision": digest(precision),
        "operators": source_fingerprint(packages)["sha256"] if packages else None,
    }


def validate_training_model(step, model, objective, data):
    """Bind a loaded joint step to a live training host before its first use.

    Supply the already prepared first local batch from ``training_batch``.
    Validation traces the explicit derivatives once, without compiling, running
    the native package, or doing label-count collectives. Revalidate if model
    configuration, trainability, objective or precision policy later changes.
    Custom EQX operators require matching installed implementation sources.
    Legacy/unverifiable packages remain usable with the standalone tensor ABI.
    """
    saved = getattr(step, "_host_binding", None)
    if not saved or "contract" not in saved:
        reason = (saved or {}).get("unsupported", "missing host contract; re-export")
        raise ValueError(f"Cannot verify joint training model: {reason}")
    from tace.models.compile.aot import as_joint_export_model
    from tace.models.compile.joint import trace_joint_training

    adapter = as_joint_export_model(model, data)
    try:
        if getattr(step, "replay_bucket", None) is not None:
            from ...targets.aoti import _bucket_trace

            trace = _bucket_trace(adapter, objective, data)
        else:
            trace = trace_joint_training(
                adapter,
                objective,
                data,
                fixed_shapes=not saved["dynamic"],
                max_edges=saved["max_edges"],
            )
        actual = _training_host_contract(model, objective, trace)
    except Exception as exc:
        raise ValueError(f"Cannot verify joint training model: {exc}") from exc
    if actual.keys() != saved["contract"].keys():
        raise ValueError(
            "Incompatible joint training host contract; re-export the package"
        )
    changed = [k for k, value in actual.items() if saved["contract"][k] != value]
    if changed:
        raise ValueError(
            f"Joint training model does not match package: {', '.join(changed)}"
        )
