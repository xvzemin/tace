"""Measured selection of complete O(2)-based CGTP evaluations."""

import warnings
from statistics import median

import torch


def select_method(
    module,
    features,
    radial,
    projection,
    amplitudes,
    edge_index,
    num_nodes,
    vectors,
    radial_network=None,
):
    """Cache a measured method for the device, shapes and derivative workload.

    Timings include geometry, radial projection, contraction and reduction.
    Gradient-enabled calls include a first reverse pass; training with vector
    gradients also includes the reverse pass of a force loss. Inputs are
    detached copies and parameter ``grad`` fields are never modified.
    """
    if torch.compiler.is_compiling():
        return module.selected_method
    if module.backend != "cuda" or not features.is_cuda or not edge_index.numel():
        return "baseline"
    with torch.cuda.device(features.device):
        if torch.cuda.is_current_stream_capturing():
            return module.selected_method

    parts = radial if isinstance(radial, tuple) else ((radial, "edge"),)
    inputs = features, projection, amplitudes, *(x for x, _ in parts), vectors
    parameters = (
        tuple(radial_network.parameters()) if radial_network is not None else ()
    )
    required = tuple(torch.is_grad_enabled() and x.requires_grad for x in inputs)
    parameter_grads = tuple(
        torch.is_grad_enabled() and p.requires_grad for p in parameters
    )
    order = (
        2 if module.training and required[-1] else int(any(required + parameter_grads))
    )
    # Neighbor counts vary between MD steps and training batches. Tune a size
    # range rather than recompiling and timing for each individual edge count.
    shapes = tuple(
        tuple(x.shape) if i == 1 else (max(0, x.size(0) - 1).bit_length(), *x.shape[1:])
        for i, x in enumerate(inputs)
    )
    key = (
        features.device,
        features.dtype,
        max(0, num_nodes - 1).bit_length(),
        max(0, edge_index.size(1) - 1).bit_length(),
        shapes,
        tuple(tuple(x.stride()) for x in inputs),
        required,
        parameter_grads,
        tuple(
            (type(layer), tuple(p.shape for p in layer.parameters(recurse=False)))
            for layer in radial_network.modules()
        )
        if radial_network is not None
        else (),
        tuple(kind for _, kind in parts),
        order,
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
    )
    if key in module.tuning_results:
        module.selected_method = module.tuning_results[key]["method"]
        return module.selected_method

    methods = ("baseline", "generator", "recurrence", "cg", "wigner")
    timings = {method: [] for method in methods}
    with (
        torch.cuda.device(features.device),
        torch.inference_mode(False),
        torch.enable_grad(),
    ):
        copied = tuple(
            x.detach().clone().requires_grad_(need) for x, need in zip(inputs, required)
        )
        differentiable = tuple(x for x in copied if x.requires_grad)
        vector_index = len(differentiable) - 1 if required[-1] else None
        differentiable += tuple(
            p for p, need in zip(parameters, parameter_grads) if need
        )

        def run(method):
            x, projection, amplitudes, *values, vectors = copied
            radial_input = (
                tuple((value, kind) for value, (_, kind) in zip(values, parts))
                if isinstance(radial, tuple)
                else values[0]
            )
            with torch.set_grad_enabled(bool(order)):
                value = module(
                    x,
                    radial_input,
                    projection,
                    None,
                    amplitudes,
                    edge_index,
                    num_nodes,
                    vectors=vectors,
                    method=method,
                    radial_network=radial_network,
                )
                results = [value]
                if order:
                    gradients = torch.autograd.grad(
                        value.square().sum(),
                        differentiable,
                        create_graph=order == 2,
                        allow_unused=True,
                    )
                    results.extend(g for g in gradients if g is not None)
                    if order == 2:
                        force = gradients[vector_index]
                        if force is not None and force.requires_grad:
                            gradients = torch.autograd.grad(
                                force.square().sum(), differentiable, allow_unused=True
                            )
                            results.extend(g for g in gradients if g is not None)
                return tuple(x.detach() for x in results)

        # Compilation and graph preparation are excluded from steady-state time.
        reference = run("baseline")
        valid = ["baseline"]
        tolerance = 5e-4 if features.dtype == torch.float32 else 5e-9
        for method in methods[1:]:
            actual = None
            try:
                actual = run(method)
                if len(actual) != len(reference):
                    raise AssertionError("Derivative outputs differ from the baseline.")
                for value, expected in zip(actual, reference):
                    torch.testing.assert_close(
                        value, expected, atol=tolerance, rtol=tolerance
                    )
                valid.append(method)
            except (torch.OutOfMemoryError, AssertionError) as error:
                warnings.warn(
                    f"O2 CGTP autotuning excluded {method}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                timings[method] = [float("inf")]
            finally:
                del actual
        del reference
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for repeat in range(3):
            for method in valid[repeat % len(valid) :] + valid[: repeat % len(valid)]:
                start.record()
                result = run(method)
                stop.record()
                stop.synchronize()
                timings[method].append(start.elapsed_time(stop))
                del result
    measured = {method: median(values) for method, values in timings.items()}
    module.selected_method = min(measured, key=measured.get)
    if len(module.tuning_results) >= 16:
        del module.tuning_results[next(iter(module.tuning_results))]
    module.tuning_results[key] = {
        "method": module.selected_method,
        "milliseconds": measured,
        "derivatives": order,
        "edges": edge_index.size(1),
    }
    return module.selected_method
