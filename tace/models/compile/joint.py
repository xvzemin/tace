"""Joint compilation of predictions, loss and parameter gradients."""

import inspect
import logging
from contextlib import nullcontext

import torch
from torch.autograd.function import once_differentiable
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

from tace.dataset.quantity import PROPERTIES
from tace.utils.loss.common import prepare_tensor_loss_counts
from tace.utils.loss.normal import NormalLoss
from tace.utils.loss.uncertainty import UncertaintyLoss

from .compile import (
    _inductor_options,
    _rebuild_graph_module,
    _remove_full_slice_scatter,
    _replace_sym_numel,
    _state_values,
    _strip_saved_tensor_detach,
)


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
        from .wrapper import _FlatE3nnCompileModel

        self.model = _FlatE3nnCompileModel(
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

    def __init__(self, model, loss_fn, data):
        from torch._decomp import get_decompositions
        from torch.fx.experimental.dynamic_spec import (
            ParamsSpec,
            ShapesSpec,
            ShapeVar,
            TensorSpec,
        )

        self.keys = tuple(
            sorted(
                key for key, value in data.items() if isinstance(value, torch.Tensor)
            )
        )
        self.module = TrainingLoss(model, loss_fn, self.keys)
        self.parameter_names = tuple(name for name, _ in self.module.named_parameters())
        self.buffer_names = tuple(name for name, _ in self.module.named_buffers())
        state = _state_values(self.module, self.parameter_names, self.buffer_names)
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
            ShapeVar("edges", min=2, optimization_hint=num_edges)
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
        edge_fields = {"edge_index": 1, "edge_shifts": 0}
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
                ("pairs", pairs),
                ("pair_edges", pair_edges),
            )
            if value is not None
        }
        for key in self.keys:
            dims = [None] * data[key].ndim
            if key in node_fields:
                dims[0] = nodes
            elif key in edge_fields:
                dims[edge_fields[key]] = edges
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
                options=dict(
                    _inductor_options(),
                    inplace_buffers=True,
                    keep_static_cubin_raw=True,
                ),
            )
            if backend == "inductor"
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
                    dynamic_shapes=ShapesSpec(ParamsSpec({"*args": specs})),
                )(
                    *[data[key].detach().contiguous().clone() for key in self.keys],
                    *state,
                )
            from equivariantx.utils.metadata import compact_metadata

            metadata_arguments = {
                "eqx.edge_program.default": (0,),
                "eqx.o3_contraction.default": (0, 1),
                "eqx.local_channel_product.default": (0, 1),
            }
            for node in graph.graph.nodes:
                positions = metadata_arguments.get(str(node.target), ())
                if positions:
                    arguments = list(node.args)
                    for position in positions:
                        if position < len(arguments) and isinstance(
                            arguments[position], str
                        ):
                            arguments[position] = compact_metadata(arguments[position])
                    node.args = tuple(arguments)
            _strip_saved_tensor_detach(graph)
            graph = _rebuild_graph_module(graph)
            _replace_sym_numel(graph)
            _remove_full_slice_scatter(graph)
            graph.graph.eliminate_dead_code()
            graph.recompile()
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
                from .metadata_comments import omit_provenance_comments

                with torch.no_grad(), omit_provenance_comments():
                    self.run = torch._inductor.standalone_compile(
                        graph,
                        [node.meta["val"].detach() for node in placeholders],
                        dynamic_shapes="from_graph",
                        options={
                            "config_patches": dict(
                                _inductor_options(),
                                inplace_buffers=True,
                                keep_static_cubin_raw=True,
                            )
                        },
                        donate_graph_module=True,
                    )
            else:
                self.run = torch.compile(
                    graph,
                    fullgraph=True,
                    dynamic=None,
                    dynamic_shapes=ShapesSpec(
                        {
                            node.name: spec
                            for node, spec in zip(placeholders, specs)
                            if spec is not None
                        }
                    ),
                    backend=backend,
                )

            if cache is not None:
                self.run = cache.save(self.run, output_schema, constants)

    def __call__(self, data):
        state = _state_values(self.module, self.parameter_names, self.buffer_names)
        inputs = [data[key].detach().contiguous() for key in self.keys]
        inputs.extend([value.detach() for value in state])
        with torch.no_grad():
            values = self.run(*inputs)
        split = 1 + len(self.trainable)
        loss, gradients, predictions = values[0], values[1:split], values[split:]
        parameters = [state[i] for i in self.trainable]
        loss = ParameterLoss.apply(loss, gradients, *parameters)
        return dict(zip(self.module.model.output_keys, predictions)), loss


def joint_training(model, loss_fn, data):
    if not supports_joint_loss(model, loss_fn):
        raise ValueError(
            "Compiled training requires PyTorch with explicit dynamic shapes, "
            "NormalLoss or UncertaintyLoss, trainable parameters and no autocast."
        )
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
    data = prepare_tensor_loss_counts(data, properties)
    key = (
        "joint",
        id(loss_fn),
        tuple(loss_fn.loss_property),
        tuple(loss_fn.loss_function_name),
        repr(loss_fn.loss_function_kwargs),
        tuple(getattr(loss_fn, "loss_property_weights", ())),
        tuple(sorted(data)),
        data["ptr"].numel(),
        min(data["positions"].shape[0], 2),
        min(data["edge_index"].shape[1], 2),
        min(data["direct_hessian_edge_pairs"].numel(), 2)
        if "direct_hessian_edge_pairs" in data
        else None,
        tuple(p.requires_grad for p in (*model.parameters(), *loss_fn.parameters())),
    )
    if key not in model._compiled_cache:
        model._compiled_cache[key] = JointTraining(model, loss_fn, data)
    return model._compiled_cache[key](data)
