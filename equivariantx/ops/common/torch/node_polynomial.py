"""Angular polynomials of a single node channel and their transposes."""

from functools import lru_cache
from itertools import permutations

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active, torch_gradient_mask


@lru_cache(maxsize=32)
def derivative_terms(degree, order, adjoint):
    """Place distinct directions in the product rule, leaving one slot free for a transpose."""
    groups = []
    for target in range(degree) if adjoint else (-1,):
        slots = tuple(i for i in range(degree) if i != target)
        terms = []
        for placement in permutations(slots, order):
            sources = dict(zip(placement, range(1, order + 1)))
            terms.append(tuple((sources.get(i, 0), i) for i in slots))
        if terms:
            groups.append((target, tuple(terms)))
    return tuple(groups)


class PolynomialMap(torch.autograd.Function):
    """Contract a directional derivative of a node polynomial, or its transpose."""

    @staticmethod
    def forward(ctx, indices, matrix, adjoint, workspace, features, seed, *directions):
        ctx.adjoint = adjoint
        ctx.workspace = workspace
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(indices, matrix, features, seed, *directions)
        degree, terms = indices.shape
        nodes, channels, components = features.shape
        width = components if adjoint else matrix.shape[1]
        output = features.new_empty((nodes, channels, width))
        groups = derivative_terms(degree, len(directions), adjoint)
        if not groups:
            return output.zero_()
        values = (features, *directions)
        if adjoint or directions:
            # The product rule reuses several tile-sized operands simultaneously.
            # On small graphs keep scratch proportional to the live node features.
            workspace = min(
                workspace, max(2 << 20, 4 * features.numel() * features.element_size())
            )
        rows = max(1, workspace // (output.element_size() * channels * terms))
        for first in range(0, nodes, rows):
            section = slice(first, first + rows)
            if not adjoint and not directions:
                x = features[section]
                product = x.index_select(-1, indices[0])
                for i in range(1, degree):
                    product.mul_(x.index_select(-1, indices[i]))
                torch.mm(
                    product.flatten(0, 1), matrix, out=output[section].flatten(0, 1)
                )
                del product
                continue
            gathered = {}
            if adjoint:
                dual = seed[section] @ matrix.T
                output[section].zero_()
            for target, products in groups:
                total = None
                for product in products:
                    factors = []
                    for source, slot in product:
                        key = source, slot
                        if key not in gathered:
                            gathered[key] = values[source][section].index_select(
                                -1, indices[slot]
                            )
                        factors.append(gathered[key])
                    term = factors[0]
                    for factor in factors[1:]:
                        term = term * factor
                    total = term if total is None else total + term
                if adjoint:
                    output[section].index_add_(-1, indices[target], total * dual)
                else:
                    torch.mm(
                        total.flatten(0, 1), matrix, out=output[section].flatten(0, 1)
                    )
        return output

    @staticmethod
    def backward(ctx, gradient):
        indices, matrix, x, seed, *directions = ctx.saved_tensors
        required = torch_gradient_mask(ctx, ctx.needs_input_grad[4:])
        gradients = [None] * len(required)
        if gradient is not None:

            def evaluate(adjoint, cotangent, vectors):
                return PolynomialMap.apply(
                    indices, matrix, adjoint, ctx.workspace, x, cotangent, *vectors
                )

            # The differential in x adds a direction; the differential in a
            # direction replaces that argument. Only the output cotangent is transposed.
            vectors = [*directions, gradient] if ctx.adjoint else directions
            cotangent = seed if ctx.adjoint else gradient
            if required[0]:
                gradients[0] = evaluate(True, cotangent, vectors)
            if ctx.adjoint and required[1]:
                gradients[1] = evaluate(False, x.new_empty(0), vectors)
            for i in range(len(directions)):
                if required[2 + i]:
                    gradients[2 + i] = evaluate(
                        True, cotangent, vectors[:i] + vectors[i + 1 :]
                    )
        return None, None, None, None, *gradients


class AngularPolynomial(torch.nn.Module):
    """Evaluate a fixed list of homogeneous polynomials on each node channel."""

    def __init__(self, columns, degree):
        super().__init__()
        terms = sorted(
            {term for p in columns for term, coefficient in p.items() if coefficient}
        )
        offsets = {term: i for i, term in enumerate(terms)}
        matrix = torch.zeros(
            len(terms), len(columns), device="cpu", dtype=torch.float64
        )
        for j, column in enumerate(columns):
            for term, coefficient in column.items():
                if coefficient:
                    matrix[offsets[term], j] = coefficient
        self.degree = degree
        self.register_buffer(
            "indices",
            torch.tensor(terms, dtype=torch.long, device="cpu")
            .reshape(-1, degree)
            .T.contiguous(),
            persistent=False,
        )
        self.master_matrix = matrix
        self.register_buffer(
            "matrix", matrix.to(torch.get_default_dtype()).clone(), persistent=False
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.matrix = self.master_matrix.to(self.matrix, copy=True)
        return self

    def forward(self, features, workspace):
        matrix = (
            self.matrix
            if self.matrix.dtype == features.dtype
            else self.master_matrix.to(features)
        )
        if self.indices.shape[1] == 0:
            return (
                features[..., :0]
                .sum(-1, keepdim=True)
                .expand(*features.shape[:-1], matrix.shape[1])
            )
        if self.degree == 1:
            return features.index_select(-1, self.indices[0]) @ matrix
        if (
            torch.compiler.is_compiling()
            or get_proxy_mode() is not None
            or is_fake(features)
            or forward_ad_active()
        ):
            product = features.index_select(-1, self.indices[0])
            for i in range(1, self.degree):
                product = product * features.index_select(-1, self.indices[i])
            return product @ matrix
        return PolynomialMap.apply(
            self.indices, matrix, False, workspace, features, features.new_empty(0)
        )
