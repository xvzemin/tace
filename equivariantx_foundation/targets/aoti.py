"""AOTInductor export and loading for EquivariantX operators."""

import json
import sys
import zipfile
from importlib import import_module

import torch

from equivariantx_foundation.registry import REGISTRATION_MODULES, require_cuda

MANIFEST = "equivariantx_namespaces.txt"
TRAINING_SCHEMA = "equivariantx_joint_training_v1"


def export(exported, output, *, inductor_configs=None):
    """Compile an exported program into a standalone AOTInductor package.

    Parameters
    ----------
    exported : torch.export.ExportedProgram
        Exported tensor program.
    output : str or pathlib.Path
        Destination ``.pt2`` file.
    inductor_configs : dict, optional
        AOTInductor compilation options.

    Examples
    --------
    >>> path = export(exported, "potential.pt2")  # doctest: +SKIP
    """
    from equivariantx.ops.compiler import compile_and_package

    devices = {
        value.device.type
        for node in exported.graph.nodes
        if isinstance(value := node.meta.get("val"), torch.Tensor)
    }
    require_cuda("cuda" if "cuda" in devices else "cpu")
    namespaces = {
        node.target.namespace
        for module in exported.graph_module.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if isinstance(node.target, torch._ops.OpOverload)
        and (node.target.namespace == "eqx" or node.target.namespace.startswith("eqx_"))
    }
    unknown = namespaces.difference({"eqx"}, REGISTRATION_MODULES)
    if unknown:
        raise ValueError(f"Unknown EquivariantX operator namespaces: {sorted(unknown)}")
    path = compile_and_package(
        exported, package_path=str(output), inductor_configs=inductor_configs
    )
    if namespaces:
        with zipfile.ZipFile(path, "a") as archive:
            root = archive.namelist()[0].split("/", 1)[0]
            archive.writestr(f"{root}/{MANIFEST}", "\n".join(sorted(namespaces)))
    return path


def _package_namespaces(archive):
    root = archive.namelist()[0].split("/", 1)[0]
    entry = f"{root}/{MANIFEST}"
    if entry in archive.namelist():
        return archive.read(entry).decode().splitlines()
    # Other exporters need not know about Foundation. Only inspect the operator
    # targets in Inductor's default model, never import a module named by it.
    prefix = f"{root}/data/aotinductor/model/"
    namespaces = set()
    for name in archive.namelist():
        if name.startswith(prefix) and name.endswith(".wrapper.json"):
            with archive.open(name) as stream:
                for extern in json.load(stream)["nodes"]:
                    namespace = extern["node"]["target"].partition("::")[0]
                    if namespace == "eqx" or namespace.startswith("eqx_"):
                        namespaces.add(namespace)
    return sorted(namespaces)


def _load_runner(path, device_index, *, single_threaded=False):
    require_cuda()
    from torch._inductor import aoti_load_package

    with zipfile.ZipFile(path) as archive:
        namespaces = _package_namespaces(archive)
    for namespace in namespaces:
        if namespace == "eqx":
            import_module("equivariantx")
        else:
            module = REGISTRATION_MODULES.get(namespace)
            if module is None:
                raise ValueError(
                    f"Unknown EquivariantX operator namespace {namespace!r}."
                )
            import_module(f"equivariantx_foundation.{module}")
    options = {"run_single_threaded": True} if single_threaded else {}
    runner = aoti_load_package(str(path), device_index=device_index, **options)
    require_cuda(runner.get_metadata()["AOTI_DEVICE_KEY"])
    return runner


class _FixedShapeRunner:
    def __init__(self, runner, metadata, device_index):
        self.compiled = runner
        self.signature = tuple(
            (key, torch.Size(shape), getattr(torch, dtype))
            for key, shape, dtype in json.loads(metadata["tace_fixed_inputs"])
        )
        device = torch.device(metadata["AOTI_DEVICE_KEY"])
        if device.type == "cuda":
            index = device.index if device_index == -1 else device_index
            device = torch.device(
                "cuda", torch.cuda.current_device() if index is None else index
            )
        self.device = device

    def get_metadata(self):
        return self.compiled.get_metadata()

    def __call__(self, *inputs):
        for (key, shape, dtype), tensor in zip(self.signature, inputs, strict=True):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.is_neg()
                or tensor.is_conj()
            ):
                raise ValueError(
                    f"Fixed AOTI input {key!r} requires a resolved, contiguous {tuple(shape)} {dtype} tensor on {self.device}."
                )
        return self.compiled(*inputs)


def load(path, *, device_index=-1, replay_inputs=None):
    """Register packaged operators and load an AOTInductor model.

    This flat tensor API does not reconstruct a TACE model or apply its graph
    input validation. For a frozen TACE graph, supply valid ``node_type``
    indices in the exported element order. A periodic charge-only LES graph
    requires all PBC flags to be true and finite cells with positive volume.

    Parameters
    ----------
    path : str or pathlib.Path
        AOTInductor package.
    device_index : int, optional
        CUDA device index. The default retains the package device.
    replay_inputs : tuple of torch.Tensor, optional
        Flat inputs for a deterministic, frozen CUDA inference package whose
        output shapes do not depend on tensor values. Capture once for these
        input shapes, dtypes and strides. Output containers are preserved.
        Later calls take new tensor values on the same device, thread and
        stream, and return independently owned outputs. Omit for dynamic shapes.

    Examples
    --------
    >>> model = load("potential.pt2")  # doctest: +SKIP
    >>> step = load("potential.pt2", replay_inputs=inputs)  # doctest: +SKIP
    >>> energy, forces, stress = step(*next_inputs)  # doctest: +SKIP
    """
    from ..execution.replay import TensorProgramReplay

    if replay_inputs is not None:
        device = TensorProgramReplay.input_device(replay_inputs)
        if device_index not in (-1, device.index):
            raise ValueError("Replay inputs must be on the selected CUDA device.")
        device_index = device.index
    runner = _load_runner(path, device_index, single_threaded=replay_inputs is not None)
    metadata = runner.get_metadata()
    if TRAINING_SCHEMA in metadata:
        raise ValueError("Use load_training for a joint training package.")
    if "tace_fixed_inputs" in metadata:
        runner = _FixedShapeRunner(runner, metadata, device_index)
    if replay_inputs is None:
        return runner
    return TensorProgramReplay(runner, replay_inputs)


def training_batch(model, objective, batch):
    """Prepare graph inputs, labels and loss counts for one joint AOTI step.

    Parameters
    ----------
    model, objective : torch.nn.Module
        Model and objective used for export.
    batch : dict
        Local labelled graph batch.

    Examples
    --------
    >>> data = training_batch(model, loss, batch)  # doctest: +SKIP
    """
    from tace.models.compile.joint import prepare_joint_labels

    from ..models.tace.training import prepare_training_data

    return prepare_joint_labels(objective, prepare_training_data(model, batch))


def training_inputs(model, objective, batch):
    """Return a prepared batch and the live parameter and buffer mappings.

    Parameters
    ----------
    model, objective : torch.nn.Module
        TACE model and objective used for export.
    batch : dict
        Local labelled graph batch.

    Examples
    --------
    >>> data, params, buffers = training_inputs(model, loss, batch)  # doctest: +SKIP
    >>> data = training_batch(model, loss, next_batch)  # doctest: +SKIP
    """
    return training_batch(model, objective, batch), *_training_state(model, objective)


def _training_state(model, objective):
    state = (model, objective)

    def named(method):
        return {
            f"{prefix}.{name}": value
            for prefix, item in zip(("model", "loss_fn"), state, strict=True)
            for name, value in getattr(item, method)()
        }

    return named("named_parameters"), named("named_buffers")


def _signature(value):
    return [
        list(value.shape),
        list(value.stride()),
        str(value.dtype),
        value.device.type,
    ]


def _static_training_graph(exported):
    if exported.range_constraints:
        return False
    unsafe = {torch.Tag.nondeterministic_seeded, torch.Tag.data_dependent_output}
    if hasattr(torch.Tag, "cudagraph_unsafe"):
        unsafe.add(torch.Tag.cudagraph_unsafe)
    for module in exported.graph_module.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                target = node.target
                if not unsafe.isdisjoint(getattr(target, "tags", ())):
                    return False
                if isinstance(
                    target, torch._ops.HigherOrderOperator
                ) and target.name() in ("cond", "while_loop"):
                    return False
    return True


def _check_stateless_training(exported):
    from torch import _decomp

    signature = exported.graph_signature
    # The initial export is training IR: add_ on a buffer (or its view)
    # is still an operator, not a USER_INPUT_MUTATION output. Functionalize
    # only graphs with possible writes; arithmetic on temporaries is legal.
    may_alias_or_mutate = getattr(_decomp, "_should_decompose_because_unsafe_op", None)
    if may_alias_or_mutate is None or any(
        isinstance(target := node.target, torch._ops.HigherOrderOperator)
        or (
            isinstance(target, torch._ops.OpOverload)
            and (target._schema.is_mutable or may_alias_or_mutate(target))
        )
        for node in exported.graph.nodes
    ):
        signature = exported.run_decompositions({}).graph_signature
    if signature.user_inputs_to_mutate or signature.buffers_to_mutate:
        raise ValueError(
            "A stateless joint package cannot mutate its inputs or buffers"
        )


def validate_training_model(step, model, objective, data):
    """Bind a loaded joint step to a live training host before its first use.

    Supply the already prepared first local batch from ``training_batch``.
    Validation traces the explicit derivatives once, without compiling, running
    the native package, or doing label-count collectives. Revalidate if model
    configuration, trainability, objective or precision policy later changes.
    Custom EQX operators require matching installed implementation sources.
    Legacy/unverifiable packages remain usable with the standalone tensor ABI.
    """
    from ..models.tace.training_contract import validate_training_model as validate

    return validate(step, model, objective, data)


def _sum_single_graph(graph, batch):
    """Replace additions indexed by a single graph ID with a sum over atoms."""
    aten = torch.ops.aten
    checked = False
    for node in list(batch.users):
        if node.target != aten.index_add.default:
            continue
        dest, dim, indices, values = node.args
        if indices is not batch or dim != 0:
            continue
        size = dest.meta["val"].shape[0]
        if (
            type(size) is not int
            or size != 1
            or not dest.meta["val"].is_floating_point()
        ):
            continue
        # A fused CUDA atomic add to a singleton can include padded reduction lanes.
        with graph.inserting_before(node):
            if not checked:
                valid = graph.call_function(aten.eq.Scalar, (batch, 0))
                valid = graph.call_function(aten.all.default, (valid,))
                graph.call_function(
                    aten._assert_async.msg, (valid, "Expected one graph.")
                )
                checked = True
            total = graph.call_function(aten.sum.dim_IntList, (values, [0], True))
            result = graph.call_function(aten.add.Tensor, (dest, total), node.kwargs)
        node.replace_all_uses_with(result)
        graph.erase_node(node)


def export_training(
    model, objective, sample, output, *, dynamic_shapes=False, max_edges=None
):
    """Package a TACE training loss and its explicit parameter gradients.

    Parameters
    ----------
    model, objective : torch.nn.Module
        Training model and loss. Parameters and buffers remain external inputs.
    sample : dict
        Labelled graph batch on the export device.
    output : str or pathlib.Path
        Destination package.
    dynamic_shapes : bool, optional
        Vary atom, edge and pair counts without changing the number of structures.
        Counts of zero or one in the sample remain fixed. Defaults to fixed shapes.
    max_edges : int, optional
        Inclusive upper bound on directed edges for a dynamic package. Inputs
        are not padded. The export sample must contain at least two edges.

    Examples
    --------
    >>> path = export_training(model, loss, batch, "train.pt2", dynamic_shapes=True)  # doctest: +SKIP
    """
    from tace.models.compile.aot import as_joint_export_model
    from tace.models.compile.compile import inductor_options
    from tace.models.compile.joint import trace_joint_training

    from ..models.tace.training_contract import _training_host_contract

    require_cuda(model)
    adapter = as_joint_export_model(model, sample)
    data = training_batch(model, objective, sample)
    if max_edges is not None and not (
        dynamic_shapes
        and type(max_edges) is int
        and max_edges > 2
        and 2 <= data["edge_index"].size(1) <= max_edges
    ):
        raise ValueError(
            "max_edges requires a dynamic training batch with 2 <= edges <= max_edges and max_edges > 2."
        )
    traced = trace_joint_training(
        adapter, objective, data, fixed_shapes=not dynamic_shapes, max_edges=max_edges
    )
    graph = traced.graph
    devices = {value.device for value in traced.args}
    if len(devices) != 1:
        raise ValueError("Joint training inputs must share one device")
    if any(value.is_conj() or value.is_neg() for value in traced.args):
        raise ValueError("Joint training requires materialized inputs.")
    try:
        binding = {
            "contract": _training_host_contract(model, objective, traced),
            "dynamic": dynamic_shapes,
            "max_edges": max_edges,
        }
    except Exception as exc:
        # Standalone inference of explicit gradients has never required a live
        # Python model; unsupported host verification must not break this ABI.
        binding = {"unsupported": str(exc)}
    (result,) = (node for node in graph.graph.nodes if node.op == "output")
    outputs = result.args[0]
    present = [value is not None for value in outputs]
    result.args = (tuple(value for value in outputs if value is not None),)
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    if dynamic_shapes and "batch" in traced.batch_keys:
        _sum_single_graph(graph.graph, placeholders[traced.batch_keys.index("batch")])
    buffer_start = len(traced.batch_keys) + len(traced.parameters)
    buffer_rows = []
    args = []
    for i, (node, value) in enumerate(zip(placeholders, traced.args, strict=True)):
        if i >= buffer_start:
            if not node.users:
                graph.graph.erase_node(node)
                continue
            buffer_rows.append(i - buffer_start)
        args.append(value)
    graph.recompile()
    dims = {}
    signatures = [_signature(value) for value in traced.args]
    shapes = None
    if dynamic_shapes:
        shapes = [{} for _ in args]
        for i, key in enumerate(traced.batch_keys):
            for axis, name in traced.dynamic_axes[key].items():
                if name not in dims:
                    dims[name] = torch.export.Dim(
                        name, min=2, max=max_edges if name == "edges" else None
                    )
                shapes[i][axis] = dims[name]
                signatures[i][0][axis] = name
    exported = torch.export.export(
        graph,
        tuple(args),
        dynamic_shapes=None if shapes is None else tuple(shapes),
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=dynamic_shapes,
    )
    if dynamic_shapes and exported._guards_code:
        raise ValueError(
            "The joint export retains Python-only shape guards that standalone AOTI "
            "cannot enforce. Use fixed shapes for this model."
        )
    _check_stateless_training(exported)
    schema = {
        "batch": traced.batch_keys,
        "parameters": traced.parameters,
        "buffers": traced.buffers,
        "gradients": traced.gradients,
        "predictions": traced.predictions,
        "present": present,
        "inputs": signatures,
        "host_binding": binding,
        "device": str(devices.pop()),
        "static_replay": not dynamic_shapes and _static_training_graph(exported),
    }
    if getattr(adapter, "_joint_local_only", False):
        schema.update(
            local_only=True,
            static_replay=False,
            forbidden_inputs=sorted(adapter._joint_forbidden_inputs),
        )
    if getattr(adapter, "_joint_positive_periodic_cell", None):
        schema["positive_periodic_cell"] = adapter._joint_positive_periodic_cell
    if dims:
        # The standalone runner does not enforce shared dimensions. Use the
        # final export's bounds, and refuse unrepresented shape specializations.
        meta = {
            n.name: n.meta["val"] for n in exported.graph.nodes if n.op == "placeholder"
        }
        symbols, limits = {}, {}
        for i, key in enumerate(traced.batch_keys):
            for axis, name in traced.dynamic_axes[key].items():
                size = meta[exported.graph_signature.user_inputs[i]].shape[axis]
                if not isinstance(size, torch.SymInt) or not size.node.expr.is_Symbol:
                    raise ValueError(
                        f"Joint export specialized dynamic axis {key}[{axis}]."
                    )
                sym = size.node.expr
                if symbols.setdefault(name, sym) != sym:
                    raise ValueError(
                        f"Joint export did not preserve the shared {name} axis."
                    )
                bounds = exported.range_constraints[sym]
                limits[name] = (
                    max(2, int(bounds.lower)),
                    int(min(sys.maxsize, bounds.upper)),
                )
        if len(set(symbols.values())) != len(symbols):
            raise ValueError(
                "Joint export equated independently varying batch dimensions."
            )
        schema["axis_limits"] = limits
    if len(buffer_rows) != len(traced.buffers):
        schema["active_buffers"] = buffer_rows
    return export(
        exported,
        output,
        inductor_configs={
            **inductor_options(shape_padding=adapter.shape_padding),
            # A raw AOTI runner does not execute ExportedProgram's Python guards.
            **(
                {"scalar_asserts": True, "unsafe_skip_scalar_range_asserts": False}
                if dynamic_shapes
                else {}
            ),
            "aot_inductor.metadata": {TRAINING_SCHEMA: json.dumps(schema)},
        },
    )


def load_training(path, *, device_index=-1, replay_inputs=None):
    """Load a stateless joint program: step(batch, params, buffers) -> predictions, loss, gradients.

    Gradients are keyed by the external parameter names and may be None for an
    unused parameter. They are first derivatives of the full training loss,
    including mixed derivatives through force/stress targets. No optimizer
    update or autograd backward is performed by this wrapper. The trainable
    parameter set is fixed at export; re-export after freezing or unfreezing
    weights. Pass materialized tensors (no lazy conjugate/negative views).
    Ordinary steps read only buffers used by the compiled program. Replay
    requires the original full buffer mapping and stable storage.

    Parameters
    ----------
    path : str or pathlib.Path
        Joint training package.
    device_index : int, optional
        Local CUDA device index.
    replay_inputs : tuple of dict, optional
        Example ``(batch, parameters, buffers)`` for fixed-shape CUDA Graph
        execution on a single thread and stream. Parameters and buffers must
        keep their storage and layout. In-place optimizer updates are visible.
        Batch values may change and each call returns fresh outputs.

    Examples
    --------
    >>> step = load_training("train.pt2", replay_inputs=(batch, params, buffers))  # doctest: +SKIP
    >>> pred, loss, grads = step(batch, params, buffers)  # doctest: +SKIP
    """
    if replay_inputs is not None:
        from ..execution.replay import TensorProgramReplay

        batch, parameters, buffers = replay_inputs
        replay_device = TensorProgramReplay.input_device((batch["positions"],))
        if device_index not in (-1, replay_device.index):
            raise ValueError("Replay inputs must be on the selected CUDA device.")
        device_index = replay_device.index
    runner = _load_runner(path, device_index, single_threaded=replay_inputs is not None)
    schema = runner.get_metadata().get(TRAINING_SCHEMA)
    if schema is None:
        raise ValueError("Not an EquivariantX joint training package")
    schema = json.loads(schema)

    def check_local_execution():
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise ValueError(
                "This joint training package supports single-rank execution only."
            )

    if schema.get("local_only"):
        check_local_execution()
    if replay_inputs is not None and not schema.get("static_replay"):
        raise ValueError(
            "Replay requires a deterministic joint export with static tensor shapes. Re-export the model or omit replay_inputs."
        )
    device = torch.device(schema["device"])
    if device.type == "cuda" and device_index >= 0:
        device = torch.device("cuda", device_index)
    count = len(schema["batch"])
    state_prefix = count + len(schema["parameters"])
    active_buffers = schema.get("active_buffers")
    buffer_names = schema["buffers"]
    input_layouts = schema["inputs"]
    if active_buffers is not None and replay_inputs is None:
        # Only these buffers reach the native program. Never look up or
        # validate thousands of unused angular tables on each ordinary step.
        buffer_names = [buffer_names[i] for i in active_buffers]
        input_layouts = [
            *input_layouts[:state_prefix],
            *(input_layouts[state_prefix + i] for i in active_buffers),
        ]
    layouts = [
        (tuple(shape), tuple(strides), getattr(torch, dtype.removeprefix("torch.")))
        for shape, strides, dtype, _ in input_layouts
    ]
    periodic_cell = tuple(
        schema["batch"].index(k) for k in schema.get("positive_periodic_cell", ())
    )
    limits = schema.get("axis_limits", {})
    varying = {
        i
        for i, (shape, _, _) in enumerate(layouts)
        if any(isinstance(n, str) for n in shape)
    }

    def state_tensors(parameters, buffers):
        yield from (parameters[key] for key in schema["parameters"])
        yield from (buffers[key] for key in buffer_names)

    def batch_tensors(batch):
        if schema.get("local_only"):
            check_local_execution()
            for key in schema["forbidden_inputs"]:
                if batch.get(key) is not None:
                    raise ValueError(
                        f"This joint training package does not accept input {key!r}."
                    )
        return [batch[key].detach().contiguous() for key in schema["batch"]]

    def arguments(batch, parameters, buffers):
        values = batch_tensors(batch)
        values.extend(state_tensors(parameters, buffers))
        extents = {}
        for i, (v, (shape, strides, dtype)) in enumerate(
            zip(values, layouts, strict=True)
        ):
            if i in varying:
                valid = v.ndim == len(shape) and v.is_contiguous()
                for n, size in zip(shape, v.shape):
                    if isinstance(n, int):
                        valid = valid and n == size
                    else:
                        lo, hi = limits[n]
                        if not lo <= size <= hi or extents.setdefault(n, size) != size:
                            raise ValueError(
                                f"Joint training input shape changed outside the shared {n} axis: "
                                f"{schema['batch'][i]} has {size}, expected {extents.get(n, (lo, hi))}"
                            )
            else:
                valid = v.shape == shape and v.stride() == strides
            if not valid or v.dtype != dtype:
                raise ValueError("Joint training input shape, stride or dtype changed")
            # Native AOTI kernels read storage directly, ignoring logical
            # conjugation/negation bits even on otherwise contiguous inputs.
            if v.is_neg() or v.is_conj():
                raise ValueError("Joint training requires materialized inputs.")
            if v.device != device:
                raise ValueError(
                    f"Joint training inputs must be on the package device {device}"
                )
        if periodic_cell:
            # A bare CUDA AOTI binary assumes these physical preconditions:
            # Inductor autotunes kernels on synthetic, nonphysical inputs.
            # Enforce them before the native call, without TACE or host sync.
            pbc, cell = (values[i] for i in periodic_cell)
            a, b, c = cell.unbind(-2)
            volume = (a * torch.linalg.cross(b, c)).sum(-1)
            torch._assert_async(
                pbc.all() & (torch.isfinite(volume) & (volume > 0)).all(),
                "Periodic cells require all PBC flags and finite positive volume.",
            )
        return values

    replay = None
    if replay_inputs is not None:
        values = arguments(batch, parameters, buffers)
        # Detached views keep the captured addresses alive, including if the
        # caller later replaces a Parameter's storage via set_ or .data.
        state = tuple(t.detach() for t in values[count:])
        # Resizing a detached alias can relocate the backing allocation of
        # *both* views without changing Tensor.is_set_to(). CUDA Graph kernels
        # retain the numerical address from capture, not the storage handle.
        addresses = tuple(t.data_ptr() for t in state)
        runtime_state = state
        if active_buffers is not None:
            num_params = len(schema["parameters"])
            runtime_state = (
                *state[:num_params],
                *(state[num_params + i] for i in active_buffers),
            )
        replay = TensorProgramReplay(
            lambda *data: runner(*data, *runtime_state), values[:count]
        )

    @torch.no_grad()
    def step(batch, parameters, buffers):
        if replay is None:
            values = arguments(batch, parameters, buffers)
            flat = iter(runner(*(t.detach() if t.requires_grad else t for t in values)))
        else:
            if any(
                t.dtype != bound.dtype
                or t.device != bound.device
                or t.is_neg()
                or t.is_conj()
                or not t.is_set_to(bound)
                or t.data_ptr() != address
                for t, bound, address in zip(
                    state_tensors(parameters, buffers), state, addresses, strict=True
                )
            ):
                raise ValueError(
                    "Joint replay parameter or buffer storage or layout changed; recapture the package."
                )
            flat = iter(replay(*batch_tensors(batch)))
        outputs = [next(flat) if exists else None for exists in schema["present"]]
        n = len(schema["gradients"])
        gradients = outputs[1 : 1 + n]
        if replay is None:
            # Inductor may return a view of an input, a constant, or the
            # same tensor for distinct parameters. Optimizers own writable
            # gradients (e.g. clip_grad_norm_); replay already clones outputs.
            live = [g for g in gradients if g is not None]
            if live:
                owned = [torch.empty_like(g) for g in live]
                torch._foreach_copy_(owned, live)
                copies = iter(owned)
                gradients = [next(copies) if g is not None else None for g in gradients]
        return (
            dict(zip(schema["predictions"], outputs[1 + n :], strict=True)),
            outputs[0],
            dict(zip(schema["gradients"], gradients, strict=True)),
        )

    step._host_binding = schema.get("host_binding")
    return step


def distributed_training_inputs(model, objective, batch):
    """Prepare a supported NormalLoss for globally averaged joint training.

    Supports the standard E/F/stress, magnetic-force and direct-Hessian MSE
    terms, and HIP's mae_energy, mae_forces and mae_direct_hessian.

    Each rank must have matching replicas (including objective weights), a
    nonempty batch of complete independent structures and a package compiled
    for its local shape. Call on *all* ranks, including those with no valid
    stress or direct-Hessian labels (use zero weights and shape-correct
    placeholders). Direct Hessians require int64 structure offsets, use packed
    ordered atom pairs, and accept either the native structure or Cartesian-
    component mean. Do not additionally divide the local loss by world size.
    This helper does not broadcast state, step an optimizer or require DDP.
    """
    import torch.distributed as dist

    from tace.utils.loss.normal import NormalLoss

    from ..models.tace.training import prepare_training_data

    # These E/F terms apply label weights *after* abs(error) or error.square().
    # Do not include Huber: weighting its residual also changes its transition.
    supported = {
        "energy": {"mse_energy_per_atom", "mse_energy", "mae_energy"},
        "forces": {"mse_forces", "mae_forces"},
        "stress": {"mse_stress", "mse_voigt_stress"},
        "noncollinear_magnetic_forces": {"mse_noncollinear_magnetic_forces"},
        "direct_hessian": {"mse_direct_hessian", "mae_direct_hessian"},
    }
    # Indices in the (structures, atoms) count vector. Each vector component
    # uses the same atom denominator, including when a label has weight zero.
    mean_count = {"energy": 0, "forces": 1, "noncollinear_magnetic_forces": 1}
    contract = None
    error = None
    if (
        type(objective) is NormalLoss
        and len(objective.loss_property) == len(objective.loss_function_name)
        and all(
            fn in supported.get(key, ())
            for key, fn in zip(objective.loss_property, objective.loss_function_name)
        )
    ):
        # Prepare locally before *any* label-count collective. A missing graph
        # field, label or weight on one rank must not strand its peers.
        try:
            data = prepare_training_data(model, batch)
            required = {
                "ptr",
                "batch",
                "node_type",
                "positions",
                "lattice",
                "edge_index",
                "edge_shifts",
                "entropy",
                *objective.loss_property,
                *(f"{k}_weight" for k in objective.loss_property),
            }
            if "noncollinear_magnetic_forces" in objective.loss_property:
                required.add("initial_noncollinear_magmoms")
            missing = required.difference(data)
            if missing:
                raise KeyError(f"missing required inputs: {sorted(missing)}")
            structures, atoms = data["ptr"].numel() - 1, len(data["positions"])
            if "direct_hessian" in objective.loss_property:
                for key, kwargs in zip(
                    objective.loss_property, objective.loss_function_kwargs
                ):
                    if key == "direct_hessian" and kwargs.get(
                        "reduction", "structure"
                    ) not in ("structure", "component"):
                        raise ValueError(
                            "direct_hessian reduction must be structure or component"
                        )
                ptr, label = data["ptr"], data["direct_hessian"]
                if ptr.ndim != 1 or ptr.dtype != torch.int64 or structures <= 0:
                    raise ValueError(
                        "direct_hessian requires an int64 ptr for nonempty structures"
                    )
                sizes = ptr.diff()
                if (
                    label.ndim != 3
                    or label.shape[1:] != (3, 3)
                    or not label.is_floating_point()
                    or not (
                        (ptr[0] == 0)
                        & (ptr[-1] == atoms)
                        & (sizes > 0).all()
                        & (sizes.square().sum() == label.shape[0])
                    ).item()
                ):
                    raise ValueError(
                        "direct_hessian labels must have shape (sum_g N_g**2, 3, 3) for complete nonempty structures"
                    )
            if (
                structures > 0
                and atoms > 0
                and data["entropy"].shape == (structures,)
                and all(
                    data[f"{k}_weight"].shape == (structures,)
                    for k in objective.loss_property
                )
                and all(
                    data[f"{k}_weight"].dtype in (torch.float32, torch.float64)
                    for k in objective.loss_property
                    if k in mean_count
                )
                and (
                    "noncollinear_magnetic_forces" not in objective.loss_property
                    or data["initial_noncollinear_magmoms"].shape
                    == data["noncollinear_magnetic_forces"].shape
                    == (atoms, 3)
                )
                and all(
                    torch.is_tensor(data[k])
                    and data[k].device == data["positions"].device
                    for k in required
                )
            ):
                params, buffers = _training_state(model, objective)
                contract = (
                    tuple(objective.loss_property),
                    tuple(objective.loss_function_name),
                    tuple(objective.loss_property_weights),
                    objective.loss_function_kwargs,
                    tuple(
                        sorted(
                            k for k, v in data.items() if isinstance(v, torch.Tensor)
                        )
                    ),
                )
        except Exception as exc:
            # Surface the actual failure to every rank; even a custom adapter's
            # local preparation can fail before its peers are ready to proceed.
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = "unsupported objective or property/loss pairing"
    world = dist.get_world_size()
    reports = [None] * world
    # The atom and structure counts are already Python shape metadata.
    sizes = (structures, atoms) if contract is not None else (0, 0)
    dist.all_gather_object(reports, (contract, error, sizes))
    failures = [
        f"rank {i}: {detail or 'missing or invalid input'}"
        for i, (s, detail, _) in enumerate(reports)
        if s is None
    ]
    if failures or any(s != contract for s, _, _ in reports):
        raise ValueError(
            "Expected matching supported NormalLoss terms, complete nonempty shards "
            "and FP32/FP64 energy/force label weights on all ranks; "
            + ("; ".join(failures) or "rank contracts differ")
        )
    from tace.models.compile.joint import prepare_joint_labels

    data = prepare_joint_labels(objective, data)
    # Stress and direct Hessians already have global valid-label denominators.
    # For a direct Hessian these counts are valid structures and sum_g N_g**2
    # over those structures (x9 for the component mean), not a squared atom total.
    # Energy/force kernels instead mean over all local structures/atoms, even
    # if an entry has zero label weight. Rescale the numerator of each term.
    for key, i in mean_count.items():
        if key in objective.loss_property:
            scale = (sizes[i] * world) / sum(
                rank_sizes[i] for _, _, rank_sizes in reports
            )
            data[f"{key}_weight"] = data[f"{key}_weight"] * scale
    return data, params, buffers


@torch.no_grad()
def mean_training_result(loss, gradients, parameters):
    """Mean rank-local loss and explicit gradients; return (loss, gradients).

    Use after distributed_training_inputs + load_training, on the default group.
    No input tensors or .grad fields are mutated. A missing local derivative
    contributes zero; an everywhere-unused parameter retains None. Names are
    canonical: tied weights must have one gradient entry, as exported by
    trace_joint_training. All ranks must call this in the same order.

    >>> data, params, buffers = distributed_training_inputs(model, objective, batch)  # doctest: +SKIP
    >>> predictions, loss, grads = step(data, params, buffers)  # doctest: +SKIP
    >>> loss, grads = mean_training_result(loss, grads, params)  # doctest: +SKIP
    >>> optimizer.zero_grad(set_to_none=True)  # doctest: +SKIP
    >>> for name, grad in grads.items():  # doctest: +SKIP
    ...     params[name].grad = grad
    >>> optimizer.step()  # doctest: +SKIP
    """
    import torch.distributed as dist

    world = dist.get_world_size()
    names, groups = _gradient_layout(loss, gradients, parameters)
    result = loss.clone()
    dist.all_reduce(result)
    return result.div_(world), _mean_gradient_blocks(
        gradients, parameters, names, groups
    )


def _gradient_layout(loss, gradients, parameters):
    from collections.abc import Mapping
    from hashlib import sha256

    import torch.distributed as dist

    mappings = isinstance(gradients, Mapping) and isinstance(parameters, Mapping)
    names = sorted(n for n in gradients if isinstance(n, str)) if mappings else []
    # Check the name ordering before any tensor collective: two different
    # packages can have identically shaped weights with different names. An
    # invalid rank must not leave its peers waiting in a gradient all-reduce.
    signature = None
    if (
        mappings
        and len(names) == len(gradients)
        and isinstance(loss, torch.Tensor)
        and loss.ndim == 0
        and loss.is_floating_point()
        and all(isinstance(p, torch.Tensor) for p in parameters.values())
        and {id(p) for p in parameters.values() if p.requires_grad}
        == {id(parameters[n]) for n in names if n in parameters}
        and len({id(parameters[n]) for n in names if n in parameters}) == len(names)
        and all(
            parameters[n].device == loss.device
            and (
                gradients[n] is None
                or (
                    isinstance(gradients[n], torch.Tensor)
                    and gradients[n].layout == torch.strided
                    and gradients[n].shape == parameters[n].shape
                    and gradients[n].dtype == parameters[n].dtype
                    and gradients[n].device == loss.device
                )
            )
            for n in names
        )
    ):
        signature = (
            str(loss.dtype),
            loss.device.type,
            [(n, tuple(parameters[n].shape), str(parameters[n].dtype)) for n in names],
        )
    # Communicate a fixed-size fingerprint, not every full parameter path.
    # The availability mask is already known on the host; do not round-trip
    # one boolean per parameter through an extra device collective + tolist().
    digest = (
        sha256(repr(signature).encode()).digest() if signature is not None else None
    )
    bits = bytearray((len(names) + 7) // 8)
    for i, n in enumerate(names):
        if gradients[n] is not None:
            bits[i // 8] |= 1 << (i % 8)
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, (digest, bytes(bits)))
    if digest is None or any(d != digest for d, _ in peers):
        raise ValueError(
            "Inconsistent distributed training names, trainable parameters, aliases or tensor signatures"
        )
    union = 0
    for _, mask in peers:
        union |= int.from_bytes(mask, "little")
    present = union.to_bytes(len(bits), "little")
    groups = {}
    for i, n in enumerate(names):
        if present[i // 8] & (1 << (i % 8)):
            groups.setdefault(parameters[n].dtype, []).append(n)
    return names, tuple(groups.values())


def _mean_gradient_blocks(gradients, parameters, names, groups):
    import torch.distributed as dist

    world = dist.get_world_size()
    mean = dict.fromkeys(names)
    for block in groups:
        packed = torch.cat(
            [
                gradients[n].reshape(-1)
                if gradients[n] is not None
                else parameters[n].new_zeros(parameters[n].numel())
                for n in block
            ]
        )
        dist.all_reduce(packed)
        chunks = packed.div_(world).split([parameters[n].numel() for n in block])
        for n, chunk in zip(block, chunks, strict=True):
            mean[n] = chunk.view_as(parameters[n])
    return mean
