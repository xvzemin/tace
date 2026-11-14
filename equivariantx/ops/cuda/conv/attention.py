"""Native segmented normalization and recomputed streaming adjoints."""

import torch

from ...common.attention import normalized_exp
from ...common.recompute import Replay


def graph_softmax(scores, target, num_nodes, weight, eps):
    """Normalize with native segmented reductions."""
    return (
        normalized_exp(scores.contiguous(), weight.contiguous(), target, num_nodes, eps)
        * weight
    )


def transpose(module, active, outputs, num_inputs):
    """Construct a bounded-memory attention adjoint with differentiable replay."""
    key = (active, outputs, num_inputs)
    if key not in module.derivatives:

        def tile_adjoint(values, create_graph):
            arguments = values[:num_inputs]
            result, denominator, maximum, grad_result, grad_denominator = values[
                num_inputs:
            ]
            nodes, angular, channels = result.shape
            heads = denominator.shape[1]
            index = arguments[module.target_slot]
            target = index[1] if index.ndim == 2 else index
            scores, value = module.edge_function(arguments)
            scores = torch.where(
                arguments[module.valid_slot][:, None], scores, -torch.inf
            )
            normalizer = arguments[module.normalizer_slot]
            value_weight = arguments[module.value_weight_slot]
            exp = (scores - maximum[target]).exp()
            quotient = exp / denominator[target]
            incoming = grad_result[target].view(-1, angular, heads, channels // heads)
            dot_value = (value.view_as(incoming) * incoming).sum((1, 3))
            dot_result = (
                (result * grad_result)
                .view(nodes, angular, heads, channels // heads)
                .sum((1, 3))[target]
            )
            grad_normalizer = (
                quotient * (value_weight * dot_value - dot_result)
                + exp * grad_denominator[target]
            )
            grad_value_weight = quotient * normalizer * dot_value
            grad_scores = grad_normalizer * normalizer
            grad_value = (
                incoming * (quotient * normalizer * value_weight)[:, None, :, None]
            ).reshape_as(value)
            pairs = [
                (x, g)
                for x, g in (
                    (scores, grad_scores),
                    (value, grad_value),
                    (normalizer, grad_normalizer.sum_to_size(normalizer.shape)),
                    (value_weight, grad_value_weight.sum_to_size(value_weight.shape)),
                )
                if x.requires_grad
            ]
            if pairs:
                result_values, vectors = zip(*pairs)
                gradients = torch.autograd.grad(
                    result_values,
                    [arguments[i] for i in active],
                    vectors,
                    create_graph=create_graph,
                    allow_unused=True,
                )
            else:
                gradients = (None,) * len(active)
            return tuple(
                (
                    torch.zeros_like(arguments[i]) if g is None else g
                    for i, g in zip(active, gradients)
                )
            )

        tile_program = Replay(tile_adjoint, active, capture=False)
        tile_program.resource = module.resource

        def adjoint(values, create_graph):
            arguments = values[:num_inputs]
            result, denominator, maximum = values[num_inputs : num_inputs + 3]
            vectors = dict(zip(outputs, values[num_inputs + 3 :]))
            grad_result = vectors.get(0, torch.zeros_like(result))
            grad_denominator = vectors.get(1, torch.zeros_like(denominator))
            shared = [None] * len(active)
            edges = [[] for _ in active]
            for chunk in module.tiles(arguments):
                gradients = tile_program(
                    *chunk, result, denominator, maximum, grad_result, grad_denominator
                )
                for slot, (i, grad) in enumerate(zip(active, gradients)):
                    if i in module.edge_axes:
                        edges[slot].append(grad)
                    else:
                        shared[slot] = (
                            grad if shared[slot] is None else shared[slot] + grad
                        )
            return tuple(
                (
                    torch.cat(edges[slot], dim=module.edge_axes[i])
                    if i in module.edge_axes
                    else shared[slot]
                    for slot, i in enumerate(active)
                )
            )

        module.derivatives[key] = Replay(adjoint, active)
        module.derivatives[key].resource = module.resource
    return module.derivatives[key]
