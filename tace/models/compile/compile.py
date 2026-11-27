################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import operator
import re
import sys
from contextlib import contextmanager
from importlib import import_module
from threading import RLock
from typing import Callable, Dict, Iterable, Sequence

import torch
from torch._decomp import get_decompositions
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

torch._dynamo.config.optimize_ddp = False
compile_comment_lock = RLock()


@contextmanager
def omit_provenance_comments():
    """Skip descriptive Triton and external-call comments during compilation."""
    modules = {
        name: import_module(name)
        for name in ("torch._inductor.codegen.triton", "torch._inductor.ir")
    }
    with compile_comment_lock:
        originals = {
            name: module.get_kernel_metadata for name, module in modules.items()
        }

        def comments_only(*args, **kwargs):
            return "", ""

        try:
            for name, module in modules.items():
                module.get_kernel_metadata = comments_only
            yield
        finally:
            for name, module in modules.items():
                module.get_kernel_metadata = originals[name]


@contextmanager
def disable_duck_shape():
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
    shape_padding: bool = True,
) -> tuple[Callable, tuple[str, ...], tuple[str, ...]]:
    parameter_names = tuple(name for name, _ in model.named_parameters())
    buffer_names = tuple(name for name, _ in model.named_buffers())
    state_values = model_state_values(model, parameter_names, buffer_names)
    state_indices = {id(value): i for i, value in enumerate(state_values)}
    state_entries = tuple(
        (f"{prefix}.{name}" if prefix else name, state_indices[id(value)])
        for prefix, module in model.named_modules()
        for name, value in (
            *module.named_parameters(recurse=False, remove_duplicate=False),
            *module.named_buffers(recurse=False, remove_duplicate=False),
        )
    )

    def compute(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        num_inputs = len(inputs)
        state = {name: args[num_inputs + index] for name, index in state_entries}
        # Write each module attribute once. Expanding shared-module aliases can
        # restore a tracing tensor instead of the original parameter or buffer.
        return functional_call(model, state, args=args[:num_inputs], tie_weights=False)

    trace_inputs = [value.clone() for value in inputs]
    trace_inputs.extend(state_values)
    traced = trace_to_fx(compute, trace_inputs)

    compiled = torch.compile(
        traced.forward,
        backend=backend,
        dynamic=True,
        fullgraph=True,
        options=inductor_options(shape_padding=shape_padding)
        if backend == "inductor"
        else None,
    )
    return compiled, parameter_names, buffer_names


def trace_to_fx(
    fn: Callable,
    inputs: Sequence[torch.Tensor],
    *,
    functionalize: bool = False,
    fixed_shapes: bool = False,
    dynamic_shapes: Sequence[dict] | None = None,
) -> torch.fx.GraphModule:
    if fixed_shapes and dynamic_shapes is not None:
        raise ValueError("A fixed trace cannot declare dynamic input dimensions.")
    mode = "fake" if fixed_shapes else "symbolic"
    with disable_duck_shape():
        traced = make_fx(
            fn,
            decomposition_table=get_decompositions(
                [torch.ops.aten.silu_backward.default]
            ),
            tracing_mode=mode,
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[value.clone() for value in inputs])
        if dynamic_shapes is not None:
            _check_frozen_cross_axis_guards(traced, inputs, dynamic_shapes)
        if functionalize:
            traced = make_fx(
                torch.func.functionalize(traced),
                tracing_mode=mode,
                _allow_non_fake_inputs=True,
                _error_on_data_dependent_ops=True,
            )(*[value.clone() for value in inputs])
            if dynamic_shapes is not None:
                _check_frozen_cross_axis_guards(traced, inputs, dynamic_shapes)
    strip_saved_tensor_detach(traced)
    traced = rebuild_graph_module(traced)
    replace_sym_numel(traced)
    remove_full_slice_scatter(traced)
    return traced


def _check_frozen_cross_axis_guards(traced, inputs, dynamic_shapes):
    """Don't package a trace specialized to a relation between independent axes.

    The subsequent export sees an FX graph, not the Python branch that produced
    it. Read *original* input sources before replacements (an observed N == E
    can already have unified the symbols in placeholder metadata). This is a
    check for guards relating distinct public size variables, not a verifier of
    single-axis control flow or arbitrary Python data-dependent decisions.
    """
    import sympy
    from torch._dynamo.source import TensorProperty, TensorPropertySource
    from torch.utils._sympy.value_ranges import ValueRanges, bound_sympy

    env = traced.shape_env
    roots, bounds = {}, {}

    def axis(dim):
        root = getattr(dim, "root", dim)
        if root not in roots:
            roots[root] = sympy.Symbol(root.__name__, integer=True, nonnegative=True)
            bounds[roots[root]] = ValueRanges(root.min, root.max)
        return roots[root] if dim is root else dim.fn(roots[root])

    shapes = [
        [axis(spec[i]) if i in spec else int(n) for i, n in enumerate(value.shape)]
        for spec, value in zip(dynamic_shapes, inputs, strict=True)
    ]
    replacements = {}
    missing = {(i, j) for i, spec in enumerate(dynamic_shapes) for j in spec}
    for symbol, sources in env.var_to_sources.items():
        for source in sources:
            if not isinstance(source, TensorPropertySource):
                continue
            name = re.fullmatch(r"input(\d+)", getattr(source.base, "source_name", ""))
            if not name:
                continue
            i = int(name.group(1))
            if i >= len(shapes):
                continue
            if source.prop is TensorProperty.SIZE:
                value = shapes[i][source.idx]
                missing.discard((i, source.idx))
            elif source.prop is TensorProperty.STRIDE and inputs[i].is_contiguous():
                value = sympy.prod(sympy.Max(1, d) for d in shapes[i][source.idx + 1 :])
            elif source.prop is TensorProperty.STORAGE_OFFSET:
                value = 0  # trace_to_fx passes clones
            else:
                continue
            if replacements.setdefault(symbol, value) != value:
                raise ValueError(
                    "Frozen trace unified independently declared input axes."
                )
    if missing and len(roots) >= 2:
        raise ValueError(
            "Cannot check independent input axes: the symbolic tracer did not "
            f"expose size sources for {sorted(missing)}. Use fixed_shapes=True."
        )
    for guard in env.guards:
        expression = guard.expr.xreplace(replacements)
        if len(expression.free_symbols & bounds.keys()) < 2:
            continue
        result = bound_sympy(expression, bounds)
        if result.lower != sympy.true or result.upper != sympy.true:
            raise ValueError(
                f"Frozen dynamic trace specialized to {expression}, not guaranteed "
                "for independent input axes. Use tensor operations for both cases "
                "or export with fixed_shapes=True."
            )


def remove_full_slice_scatter(graph_module: torch.fx.GraphModule) -> None:
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
        *model_state_values(model, parameter_names, buffer_names),
    )


def model_state_values(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
    buffer_names: Sequence[str],
) -> list[torch.Tensor]:
    parameters: Dict[str, torch.Tensor] = dict(model.named_parameters())
    buffers: Dict[str, torch.Tensor] = dict(model.named_buffers())
    return [parameters[name] for name in parameter_names] + [
        buffers[name] for name in buffer_names
    ]


def inductor_options(*, shape_padding: bool = True) -> Dict[str, object]:
    options: Dict[str, object] = {
        "max_autotune": False,
        "shape_padding": shape_padding,
        "epilogue_fusion": False,
        # In-place buffer reuse can corrupt parameter gradients through forces.
        "inplace_buffers": False,
        "triton.cudagraphs": False,
        "max_fusion_size": 8,
        "triton.persistent_reductions": False,
        # Mixed-order fusion guards on the workload size (e.g. 64 * edges
        # crossing 5 Mi elements), recompiling force-training graphs per rank.
        "triton.mix_order_reduction": False,
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


def replace_sym_numel(graph_module: torch.fx.GraphModule) -> None:
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


def strip_saved_tensor_detach(graph_module: torch.fx.GraphModule) -> None:
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


def rebuild_graph_module(
    graph_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    node_map: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in graph_module.graph.nodes:
        node_map[node] = graph.node_copy(node, lambda old: node_map[old])
    graph.lint()
    return torch.fx.GraphModule(graph_module, graph)
