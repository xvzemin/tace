################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import inspect
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from importlib import import_module

import torch
from torch.autograd.function import once_differentiable
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

from tace.dataset.quantity import PROPERTIES
from tace.utils.loss.common import prepare_tensor_loss_counts
from tace.utils.loss.normal import NormalLoss
from tace.utils.loss.registry import LOSS_FN, LOSS_MODULES
from tace.utils.loss.uncertainty import UncertaintyLoss

from .compile import (
    inductor_options,
    model_state_values,
    omit_provenance_comments,
    rebuild_graph_module,
    remove_full_slice_scatter,
    replace_sym_numel,
    strip_saved_tensor_detach,
)


@cache
def _builtin_loss_functions():
    functions = {}
    for name in LOSS_MODULES:
        module = import_module(f"tace.utils.loss.{name}")
        functions.update(
            (key, fn)
            for key, fn in vars(module).items()
            if getattr(fn, "__module__", None) == module.__name__
            and hasattr(fn, "__wrapped__")
        )
    return functions


def _cacheable_loss(loss_fn):
    if type(loss_fn) not in (NormalLoss, UncertaintyLoss):
        return False
    native = _builtin_loss_functions()
    for name in loss_fn.loss_function_name:
        if not loss_fn.training:
            name = name.replace("_dens_", "_")
        if name not in native or LOSS_FN.get(name) is not native[name]:
            return False
    return True


def supports_joint_loss(model, loss_fn):
    """Whether the loss can expose its trainable state to the joint program."""
    return (
        type(loss_fn) in (NormalLoss, UncertaintyLoss)
        and "dynamic_shapes" in inspect.signature(make_fx).parameters
        and "dynamic_shapes" in inspect.signature(torch.compile).parameters
        and hasattr(torch._inductor, "standalone_compile")
        and not torch.is_autocast_enabled("cuda")
        and not torch.is_autocast_enabled("cpu")
        and any(p.requires_grad for p in (*model.parameters(), *loss_fn.parameters()))
    )


class ParameterLoss(torch.autograd.Function):
    """Attach precomputed first parameter derivatives to a scalar loss."""

    @staticmethod
    def forward(ctx, loss, gradients, *parameters):
        ctx.save_for_backward(*gradients)
        return loss.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        # Multiplication gives each parameter independent storage even if the
        # compiler shared outputs. It also applies Lightning's loss scaling.
        gradients = ctx.saved_tensors
        result = [None] * len(gradients)
        groups = {}
        for i, gradient in enumerate(gradients):
            if gradient is not None:
                groups.setdefault((gradient.device, gradient.dtype), []).append(i)
        for (device, dtype), indices in groups.items():
            values = torch._foreach_mul(
                [gradients[i] for i in indices], scale.to(device=device, dtype=dtype)
            )
            for i, value in zip(indices, values):
                result[i] = value
        return None, None, *result


class TrainingLoss(torch.nn.Module):
    def __init__(self, model, loss_fn, input_keys):
        super().__init__()
        from .wrapper import FlatTorchModel

        self.model = FlatTorchModel(
            model, model._input_keys(input_keys), model._output_keys()
        )
        self.loss_fn = loss_fn
        self.input_keys = tuple(input_keys)

    def forward(self, *args):
        data = dict(zip(self.input_keys, args))
        predictions = dict(
            zip(
                self.model.output_keys,
                self.model(*(data[key] for key in self.model.input_keys)),
            )
        )
        return self.loss_fn(predictions, data), tuple(predictions.values())


@dataclass(frozen=True)
class JointTrainingTrace:
    graph: torch.fx.GraphModule
    args: tuple
    batch_keys: tuple
    parameters: tuple
    buffers: tuple
    gradients: tuple
    predictions: tuple
    dynamic_axes: dict[str, dict[int, str]]


class JointTraining:
    """Compile a training loss and its first parameter derivatives.

    Parameters
    ----------
    model : CompileTensorModel
        Training model. Node and edge counts may vary between calls.
    loss_fn : NormalLoss or UncertaintyLoss
        Property losses, including label weights.
    data : dict
        Prepared graph and labels. Structure count is static per program.
    """

    def __init__(
        self,
        model,
        loss_fn,
        data,
        *,
        _trace_only=False,
        _fixed_shapes=False,
        _max_edges=None,
    ):
        from torch._decomp import get_decompositions
        from torch.fx.experimental.dynamic_spec import (
            ParamsSpec,
            ShapesSpec,
            ShapeVar,
            TensorSpec,
        )

        if not _trace_only and not _cacheable_loss(loss_fn):
            raise ValueError(
                "Compiled joint training requires unchanged built-in loss functions"
            )
        self.keys = tuple(
            sorted(
                key for key, value in data.items() if isinstance(value, torch.Tensor)
            )
        )
        self.module = TrainingLoss(model, loss_fn, self.keys)
        self.parameter_names = tuple(name for name, _ in self.module.named_parameters())
        self.buffer_names = tuple(name for name, _ in self.module.named_buffers())
        state = model_state_values(self.module, self.parameter_names, self.buffer_names)
        indices = {id(value): i for i, value in enumerate(state)}
        entries = tuple(
            (f"{prefix}.{name}" if prefix else name, indices[id(value)])
            for prefix, module in self.module.named_modules()
            for name, value in (
                *module.named_parameters(recurse=False, remove_duplicate=False),
                *module.named_buffers(recurse=False, remove_duplicate=False),
            )
        )
        n = len(self.keys)
        trainable = tuple(
            i for i, p in enumerate(self.module.parameters()) if p.requires_grad
        )
        self.trainable = trainable

        def compute(*args):
            loss, predictions = functional_call(
                self.module,
                {name: args[n + index] for name, index in entries},
                args=args[:n],
                tie_weights=False,
            )
            gradients = torch.autograd.grad(
                loss, tuple(args[n + i] for i in trainable), allow_unused=True
            )
            return (
                loss.detach(),
                *(g.detach() if g is not None else None for g in gradients),
                *(p.detach() for p in predictions),
            )

        # Size hints guide scheduling without fixing dynamic graph dimensions.
        num_nodes, num_edges = data["positions"].shape[0], data["edge_index"].shape[1]
        nodes = (
            ShapeVar("nodes", min=2, optimization_hint=num_nodes)
            if num_nodes > 1
            else None
        )
        edges = (
            ShapeVar("edges", min=2, max=_max_edges, optimization_hint=num_edges)
            if num_edges > 1
            else None
        )
        node_fields = {
            "positions",
            "atomic_numbers",
            "node_type",
            "batch",
            "num_neighbors",
            "forces",
            "direct_forces",
        }
        node_fields.update(
            key for key, spec in PROPERTIES.items() if spec["scope"] == "per-atom"
        )
        edge_fields = {
            "edge_index": 1,
            "edge_shifts": 0,
            "edge_vector": 0,
            "edge_pair_inverse": 0,
        }
        edge_fields.update(
            (key, 0) for key, spec in PROPERTIES.items() if spec["scope"] == "per-edge"
        )
        bond_rows = None
        if "edge_pair_rows" in data and data["edge_pair_rows"].numel() > 1:
            bond_rows = ShapeVar(
                "undirected_edges",
                min=2,
                optimization_hint=data["edge_pair_rows"].numel(),
            )
        pair_fields = {"direct_hessian": 0, "direct_hessian_pair_index": 1}
        pair_edge_fields = {"direct_hessian_edge_pairs", "direct_hessian_edge_shifts"}
        pairs = pair_edges = None
        if "direct_hessian_pair_index" in data:
            num_pairs = data["direct_hessian_pair_index"].shape[1]
            num_pair_edges = data["direct_hessian_edge_pairs"].numel()
            pairs = (
                ShapeVar("pairs", min=2, optimization_hint=num_pairs)
                if num_pairs > 1
                else None
            )
            pair_edges = (
                ShapeVar("pair_edges", min=2, optimization_hint=num_pair_edges)
                if num_pair_edges > 1
                else None
            )
        specs = []
        dynamic_axes = {}
        axis_names = {
            id(value): name
            for name, value in (
                ("nodes", nodes),
                ("edges", edges),
                ("undirected_edges", bond_rows),
                ("pairs", pairs),
                ("pair_edges", pair_edges),
            )
            if value is not None
        }
        for key in self.keys:
            if _fixed_shapes:
                specs.append(None)
                continue
            dims = [None] * data[key].ndim
            if key in node_fields:
                dims[0] = nodes
            elif key in edge_fields:
                dims[edge_fields[key]] = edges
            elif key == "edge_pair_rows":
                dims[0] = bond_rows
            elif key in pair_fields:
                dims[pair_fields[key]] = pairs
            elif key in pair_edge_fields:
                dims[0] = pair_edges
            dynamic_axes[key] = {
                i: axis_names[id(dim)] for i, dim in enumerate(dims) if dim is not None
            }
            specs.append(TensorSpec(dims))
        specs.extend([None] * len(state))
        from .artifact import JointArtifactCache

        backend = getattr(model.readout_fn, "compile_backend", "inductor")
        compile_options = (
            dict(
                inductor_options(shape_padding=model.shape_padding),
                inplace_buffers=True,
                keep_static_cubin_raw=True,
            )
            if backend == "inductor" and not _trace_only
            else None
        )
        runtime_values = tuple(
            data[key].detach().contiguous() for key in self.keys
        ) + tuple(value.detach() for value in state)
        cache = (
            JointArtifactCache.create(
                module=self.module,
                keys=self.keys,
                parameter_names=self.parameter_names,
                buffer_names=self.buffer_names,
                trainable=self.trainable,
                entries=entries,
                values=runtime_values,
                dynamic_axes=dynamic_axes,
                options=compile_options,
            )
            if backend == "inductor" and not _trace_only
            else None
        )
        self.artifact_info = cache.info if cache is not None else {"status": "disabled"}
        with cache.transaction() if cache is not None else nullcontext():
            if cache is not None:
                loaded = cache.load()
                if loaded is not None:
                    self.run = loaded
                    return
            with torch.enable_grad():
                graph = make_fx(
                    compute,
                    decomposition_table=get_decompositions(
                        [
                            torch.ops.aten.silu_backward.default,
                            torch.ops.aten.constant_pad_nd.default,
                        ]
                    ),
                    tracing_mode="fake",
                    _allow_non_fake_inputs=True,
                    _error_on_data_dependent_ops=True,
                    dynamic_shapes=None
                    if _fixed_shapes
                    else ShapesSpec(ParamsSpec({"*args": specs})),
                )(
                    *[data[key].detach().contiguous().clone() for key in self.keys],
                    *state,
                )
            strip_saved_tensor_detach(graph)
            graph = rebuild_graph_module(graph)
            replace_sym_numel(graph)
            remove_full_slice_scatter(graph)
            graph.graph.eliminate_dead_code()
            graph.recompile()
            if _trace_only:
                self.trace = JointTrainingTrace(
                    graph,
                    runtime_values,
                    self.keys,
                    self.parameter_names,
                    self.buffer_names,
                    tuple(self.parameter_names[i] for i in trainable),
                    tuple(self.module.model.output_keys),
                    dynamic_axes,
                )
                return
            constants = ()
            if cache is not None:
                try:
                    graph, constants = cache.lift_constants(graph, runtime_values)
                except (TypeError, ValueError) as error:
                    cache.info.update(status="unsupported", reason=str(error))
                    logging.warning(
                        "Joint constant persistence unsupported; original compilation retained: %s",
                        error,
                    )
                    cache = None
            logging.info(
                "Compiling joint training graph: %d operations", len(graph.graph.nodes)
            )
            placeholders = [
                node for node in graph.graph.nodes if node.op == "placeholder"
            ]
            output_node = next(
                node for node in graph.graph.nodes if node.op == "output"
            )
            output_schema = {
                "none_mask": [value is None for value in output_node.args[0]]
            }
            # Derivatives are explicit here; no later backward needs these buffers.
            if backend == "inductor":
                with torch.no_grad(), omit_provenance_comments():
                    self.run = torch._inductor.standalone_compile(
                        graph,
                        [node.meta["val"].detach() for node in placeholders],
                        dynamic_shapes="from_graph",
                        options={"config_patches": compile_options},
                        donate_graph_module=True,
                    )
            else:
                self.run = torch.compile(
                    graph,
                    fullgraph=True,
                    dynamic=False if _fixed_shapes else None,
                    dynamic_shapes=None
                    if _fixed_shapes
                    else ShapesSpec(
                        {
                            node.name: spec
                            for node, spec in zip(placeholders, specs)
                            if spec is not None
                        }
                    ),
                    backend=backend,
                )

            compiled = self.run
            if cache is not None:
                self.run = cache.save(compiled, output_schema, constants)
            # Artifact wrappers already enforce all static dimensions.
            if _fixed_shapes and self.run is compiled:
                shapes = tuple(data[key].shape for key in self.keys)

                def run_fixed(*args):
                    if any(arg.shape != shape for arg, shape in zip(args, shapes)):
                        raise ValueError("Fixed joint training input shape changed")
                    return compiled(*args)

                self.run = run_fixed

    def __call__(self, data, *, return_gradients=False):
        state = model_state_values(self.module, self.parameter_names, self.buffer_names)
        inputs = [data[key].detach().contiguous() for key in self.keys]
        inputs.extend([value.detach() for value in state])
        with torch.no_grad():
            values = self.run(*inputs)
        split = 1 + len(self.trainable)
        loss, gradients, predictions = values[0], values[1:split], values[split:]
        parameters = [state[i] for i in self.trainable]
        outputs = dict(zip(self.module.model.output_keys, predictions))
        if return_gradients:
            return outputs, loss, zip(parameters, gradients, strict=True)
        loss = ParameterLoss.apply(loss, gradients, *parameters)
        return outputs, loss


def prepare_joint_labels(loss_fn, data):
    """Prepare the same loss-count tensors for native and packaged joint programs."""
    properties = (
        loss_fn.loss_property
        if isinstance(loss_fn, UncertaintyLoss)
        else [
            key
            for key in loss_fn.loss_property
            if key
            in (
                "stress",
                "virials",
                "direct_stress",
                "direct_virials",
                "direct_hessian",
            )
        ]
    )
    return prepare_tensor_loss_counts(data, properties)


def trace_joint_training(model, loss_fn, data, *, fixed_shapes=False, max_edges=None):
    """Trace the explicit loss, predictions and parameter derivatives without compiling.

    All parameters and buffers are positional inputs, in the same order as
    regular joint training. The caller supplies a prepared, labeled batch.
    Set ``fixed_shapes=True`` when the consumer accepts only this batch layout.
    """
    if (
        not model.training
        or not loss_fn.training
        or not supports_joint_loss(model, loss_fn)
    ):
        raise ValueError(
            "Expected a training-mode compile adapter and a supported joint loss"
        )
    # CUDA autograd can interleave independent LES/stress derivative branches
    # differently on repeated traces. Keep the materialized graph (and its
    # strict ownership fingerprint) deterministic, without changing execution.
    with (
        torch.autograd.set_multithreading_enabled(False)
        if getattr(model, "_joint_serial_trace", False)
        else nullcontext()
    ):
        return JointTraining(
            model,
            loss_fn,
            data,
            _trace_only=True,
            _fixed_shapes=fixed_shapes,
            _max_edges=max_edges,
        ).trace


def joint_training(model, loss_fn, data, *, return_gradients=False):
    if not _cacheable_loss(loss_fn):
        if return_gradients:
            raise ValueError(
                "Compiled joint training requires unchanged built-in loss functions"
            )
        from tace.models.adapter import TensorModel

        output = TensorModel.forward(model, data)
        return output, loss_fn(output, data)
    if not supports_joint_loss(model, loss_fn):
        raise ValueError(
            "Compiled training requires PyTorch with explicit dynamic shapes, "
            "NormalLoss or UncertaintyLoss, trainable parameters and no autocast."
        )
    data = prepare_joint_labels(loss_fn, data)
    key = (
        "joint",
        id(loss_fn),
        loss_fn.training,
        tuple(loss_fn.loss_property),
        tuple(loss_fn.loss_function_name),
        repr(loss_fn.loss_function_kwargs),
        tuple(getattr(loss_fn, "loss_property_weights", ())),
        tuple(sorted(data)),
        data["ptr"].numel(),
        min(data["positions"].shape[0], 2),
        min(data["edge_index"].shape[1], 2),
        min(data["edge_pair_rows"].numel(), 2) if "edge_pair_rows" in data else None,
        min(data["direct_hessian_edge_pairs"].numel(), 2)
        if "direct_hessian_edge_pairs" in data
        else None,
        tuple(p.requires_grad for p in (*model.parameters(), *loss_fn.parameters())),
    )
    if key not in model._compiled_cache:
        model._compiled_cache[key] = JointTraining(model, loss_fn, data)
    return model._compiled_cache[key](data, return_gradients=return_gradients)
