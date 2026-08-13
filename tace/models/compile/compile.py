import operator
import sys
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Sequence

import torch
from torch._decomp import get_decompositions
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

torch._dynamo.config.optimize_ddp = False


@contextmanager
def _disable_duck_shape():
    previous = torch.fx.experimental._config.use_duck_shape
    torch.fx.experimental._config.use_duck_shape = False
    try:
        yield
    finally:
        torch.fx.experimental._config.use_duck_shape = previous


def trace_and_compile(
    model: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    *,
    backend: str,
) -> tuple[Callable, tuple[str, ...], tuple[str, ...]]:
    parameter_names = tuple(name for name, _ in model.named_parameters())
    buffer_names = tuple(name for name, _ in model.named_buffers())
    state_names = parameter_names + buffer_names

    def compute(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        num_inputs = len(inputs)
        state = {name: value for name, value in zip(state_names, args[num_inputs:])}
        return functional_call(model, state, args=args[:num_inputs])

    trace_inputs = [value.clone() for value in inputs]
    trace_inputs.extend(_state_values(model, parameter_names, buffer_names))
    traced = trace_to_fx(compute, trace_inputs)

    compiled = torch.compile(
        traced,
        backend=backend,
        dynamic=True,
        fullgraph=True,
        options=_inductor_options() if backend == "inductor" else None,
    )
    return compiled, parameter_names, buffer_names


def trace_to_fx(
    fn: Callable,
    inputs: Sequence[torch.Tensor],
    *,
    functionalize: bool = False,
) -> torch.fx.GraphModule:
    with _disable_duck_shape():
        traced = make_fx(
            fn,
            decomposition_table=get_decompositions(
                [torch.ops.aten.silu_backward.default]
            ),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[value.clone() for value in inputs])
        if functionalize:
            traced = make_fx(
                torch.func.functionalize(traced),
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
                _error_on_data_dependent_ops=True,
            )(*[value.clone() for value in inputs])
    _strip_saved_tensor_detach(traced)
    traced = _rebuild_graph_module(traced)
    _replace_sym_numel(traced)
    _remove_full_slice_scatter(traced)
    return traced


def _remove_full_slice_scatter(graph_module: torch.fx.GraphModule) -> None:
    """Remove identity slice scatters emitted for a full ``:`` assignment."""
    changed = False
    for node in list(graph_module.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.slice_scatter.default
        ):
            continue
        start = node.args[3] if len(node.args) > 3 else node.kwargs.get("start")
        end = node.args[4] if len(node.args) > 4 else node.kwargs.get("end")
        step = node.args[5] if len(node.args) > 5 else node.kwargs.get("step", 1)
        if start not in {None, 0} or end not in {None, sys.maxsize} or step != 1:
            continue
        node.replace_all_uses_with(node.args[1])
        graph_module.graph.erase_node(node)
        changed = True
    if changed:
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()


def compiled_call(
    compiled: Callable,
    model: torch.nn.Module,
    inputs: Iterable[torch.Tensor],
    parameter_names: Sequence[str],
    buffer_names: Sequence[str],
) -> tuple[torch.Tensor, ...]:
    return compiled(
        *inputs,
        *_state_values(model, parameter_names, buffer_names),
    )


def _state_values(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
    buffer_names: Sequence[str],
) -> list[torch.Tensor]:
    parameters: Dict[str, torch.Tensor] = dict(model.named_parameters())
    buffers: Dict[str, torch.Tensor] = dict(model.named_buffers())
    return [parameters[name] for name in parameter_names] + [
        buffers[name] for name in buffer_names
    ]


def _inductor_options() -> Dict[str, object]:
    options: Dict[str, object] = {
        "max_autotune": False,
        "shape_padding": True,
        "epilogue_fusion": False,
        "triton.cudagraphs": False,
        "max_fusion_size": 8,
        "triton.persistent_reductions": False,
        "triton.max_tiles": 1,
    }
    try:
        from torch._inductor import config

        valid_options = config.get_config_copy()
        return {
            key: value
            for key, value in options.items()
            if key.replace("-", "_") in valid_options
        }
    except Exception:
        return options


def _replace_sym_numel(graph_module: torch.fx.GraphModule) -> None:
    graph = graph_module.graph
    for node in list(graph.nodes):
        if node.target != torch.ops.aten.sym_numel.default:
            continue
        tensor = node.args[0]
        rank = len(tensor.meta["val"].shape)
        product = 1
        with graph.inserting_before(node):
            for dim in range(rank):
                size = graph.call_function(
                    torch.ops.aten.sym_size.int,
                    args=(tensor, dim),
                )
                product = graph.call_function(operator.mul, args=(product, size))
        node.replace_all_uses_with(product)
        graph.erase_node(node)
    graph.lint()
    graph_module.recompile()


def _strip_saved_tensor_detach(graph_module: torch.fx.GraphModule) -> None:
    detach = torch.ops.aten.detach.default

    def is_detach(node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target == detach

    to_remove = []
    for node in graph_module.graph.nodes:
        if not is_detach(node):
            continue
        users = list(node.users)
        if (
            is_detach(node.args[0])
            or not users
            or all(is_detach(user) for user in users)
        ):
            to_remove.append(node)

    for node in to_remove:
        node.replace_all_uses_with(node.args[0])
        graph_module.graph.erase_node(node)
    graph_module.graph.lint()
    graph_module.recompile()


def _rebuild_graph_module(
    graph_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    node_map: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in graph_module.graph.nodes:
        node_map[node] = graph.node_copy(node, lambda old: node_map[old])
    graph.lint()
    return torch.fx.GraphModule(graph_module, graph)
