"""Static edge expressions and recursively generated analytic adjoints."""

from functools import lru_cache

from ..utils.metadata import parse_metadata


def activate(program, value, module):
    """Encode a supported scalar activation and its normalization."""
    if module is None:
        return value
    name = type(module).__name__
    if name == "normalize2mom":
        return program.scale(
            activate(program, value, module.f), 1 if module._is_id else module.cst
        )
    if name == "ScaledActivation":
        return program.scale(
            activate(program, value, module.activation), module.scale_factor
        )
    if name in ("ScaledSiLU", "ScaledSigmoid", "ScaledTanh"):
        return program.scale(
            program.unary(
                {
                    "ScaledSiLU": "silu",
                    "ScaledSigmoid": "sigmoid",
                    "ScaledTanh": "tanh",
                }[name],
                value,
            ),
            module.scale_factor,
        )
    operations = {
        "SiLU": "silu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
        "Identity": "identity",
        "silu": "silu",
        "tanh": "tanh",
        "sigmoid": "sigmoid",
    }
    if hasattr(module, "__name__"):
        name = module.__name__
    if name not in operations:
        raise NotImplementedError(f"Fused activation does not support {name}.")
    return value if name == "Identity" else program.unary(operations[name], value)


class Program:
    """A fixed-width expression graph for fused edge operations.

    Parameters and node features remain external arrays. Intermediate vectors
    have a fixed width independent of the number of edges. Reverse programs
    use the same operations as the forward, including nonlinear derivatives.
    """

    def __init__(self, nodes=()):
        self.nodes = list(nodes)
        self.cache = {node: i for i, node in enumerate(nodes)}

    def add(self, op, size, args=(), data=()):
        node = (op, size, tuple(args), data)
        if node not in self.cache:
            self.cache[node] = len(self.nodes)
            self.nodes.append(node)
        return self.cache[node]

    def size(self, value):
        return self.nodes[value][1]

    def input(self, slot, kind, size):
        return self.add("input", size, data=(slot, kind))

    def constant(self, size, value):
        return self.add("constant", size, data=float(value))

    def unary(self, op, x):
        return self.add(op, self.size(x), (x,))

    def binary(self, op, x, y):
        if self.size(x) != self.size(y):
            raise ValueError("Edge expressions must have equal vector widths.")
        return self.add(op, self.size(x), (x, y))

    def scale(self, x, value):
        return (
            x
            if value == 1
            else self.binary("mul", x, self.constant(self.size(x), value))
        )

    def sum(self, values, size):
        if not values:
            return self.constant(size, 0)
        result = values[0]
        for x in values[1:]:
            result = self.binary("add", result, x)
        return result

    def gather(self, x, indices):
        indices = tuple(indices)
        if indices == tuple(range(self.size(x))):
            return x
        return self.add("gather", len(indices), (x,), indices)

    def scatter(self, x, indices, size):
        return self.add("scatter", size, (x,), tuple(indices))

    def slice(self, x, start, size):
        if start == 0 and size == self.size(x):
            return x
        return self.add("slice", size, (x,), start)

    def concatenate(self, values):
        if len(values) == 1:
            return values[0]
        return self.add("concat", sum(self.size(x) for x in values), values)

    def transpose(self, x, rows, columns):
        return (
            x
            if rows == 1 or columns == 1
            else self.add("transpose", rows * columns, (x,), (rows, columns))
        )

    def matmul(self, x, weight, rows, inputs, outputs):
        if rows == 1 and self.nodes[x][0] == "concat":
            parts, offset = [], 0
            for value in self.nodes[x][2]:
                width = self.size(value)
                parts.append(
                    self.matmul(
                        value,
                        self.slice(weight, offset * outputs, width * outputs),
                        1,
                        width,
                        outputs,
                    )
                )
                offset += width
            return self.sum(parts, outputs)
        return self.add("matmul", rows * outputs, (x, weight), (rows, inputs, outputs))

    def product(self, values, metadata, role, required=0):
        width, dims, _ = metadata
        return self.add(
            "product", width * dims[role], values, (metadata, role, required)
        )

    def adjoint(self, outputs, seeds, active):
        """Differentiate a vector program, without evaluating tensor operations."""
        count = len(self.nodes)
        needed = []
        for op, _, args, data in self.nodes[:count]:
            needed.append(
                data[0] in active if op == "input" else any(needed[i] for i in args)
            )
        gradients = {}

        def accumulate(index, value):
            if needed[index]:
                gradients.setdefault(index, []).append(value)

        for output, seed in zip(outputs, seeds):
            accumulate(output, seed)
        results = []
        for i in reversed(range(count)):
            if i not in gradients:
                continue
            op, size, args, data = self.nodes[i]
            grad = self.sum(gradients[i], size)
            if op == "input":
                slot, kind = data
                results.append((grad, slot, kind))
                continue
            if op == "constant":
                continue
            if op == "add":
                for x in args:
                    accumulate(x, grad)
            elif op == "mul":
                x, y = args
                accumulate(x, self.binary("mul", grad, y))
                accumulate(y, self.binary("mul", grad, x))
            elif op == "convolution":
                kind, metadata, output, weighted, shape, order = data
                geometric = kind == "o3" and len(parse_metadata(metadata)) > 3
                values = list(args)
                values[output] = grad
                weighted = weighted or output in (1, 2)
                for role, x in enumerate(args):
                    if role == output or not needed[x] or not self.size(x):
                        continue
                    operands, result = values, role
                    if geometric and role == 5:
                        operands = values + [x]
                        result = len(values)
                    accumulate(
                        x,
                        self.add(
                            "convolution",
                            self.size(x),
                            operands,
                            (kind, metadata, result, weighted, shape, order + 1),
                        ),
                    )
            elif op in ("normalize", "inv_norm"):
                x = args[0]
                inverse = self.add("inv_norm", size, (x,), data)
                normalized = self.add("normalize", size, (x,), data)
                if op == "inv_norm":
                    factor = self.binary("mul", inverse, inverse)
                    factor = self.binary("mul", factor, normalized)
                    dx = self.scale(
                        self.binary("mul", self.unary("mean", grad), factor), -1
                    )
                else:
                    dot = self.unary("mean", self.binary("mul", grad, normalized))
                    dx = self.binary(
                        "add", grad, self.scale(self.binary("mul", normalized, dot), -1)
                    )
                    if data[1]:
                        dx = self.binary(
                            "add", dx, self.scale(self.unary("mean", grad), -1)
                        )
                    dx = self.binary("mul", dx, inverse)
                accumulate(x, dx)
            elif op == "mean":
                accumulate(args[0], self.unary("mean", grad))
            elif op == "rsqrt":
                cube = self.binary("mul", self.binary("mul", i, i), i)
                accumulate(args[0], self.binary("mul", grad, self.scale(cube, -0.5)))
            elif op == "reciprocal":
                factor = self.scale(self.binary("mul", i, i), -1)
                accumulate(args[0], self.binary("mul", grad, factor))
            elif op in ("exp", "sin", "cos", "tanh", "sigmoid", "silu"):
                x = args[0]
                one = self.constant(size, 1)
                if op == "exp":
                    factor = i
                elif op == "sin":
                    factor = self.unary("cos", x)
                elif op == "cos":
                    factor = self.scale(self.unary("sin", x), -1)
                elif op == "tanh":
                    factor = self.binary(
                        "add", one, self.scale(self.binary("mul", i, i), -1)
                    )
                else:
                    sigmoid = i if op == "sigmoid" else self.unary("sigmoid", x)
                    complement = self.binary("add", one, self.scale(sigmoid, -1))
                    factor = self.binary("mul", sigmoid, complement)
                    if op == "silu":
                        factor = self.binary(
                            "add", sigmoid, self.binary("mul", x, factor)
                        )
                accumulate(x, self.binary("mul", grad, factor))
            elif op == "slice":
                accumulate(
                    args[0], self.add("unslice", self.size(args[0]), (grad,), data)
                )
            elif op == "unslice":
                accumulate(args[0], self.slice(grad, data, self.size(args[0])))
            elif op == "concat":
                start = 0
                for x in args:
                    accumulate(x, self.slice(grad, start, self.size(x)))
                    start += self.size(x)
            elif op == "gather":
                accumulate(args[0], self.scatter(grad, data, self.size(args[0])))
            elif op == "scatter":
                accumulate(args[0], self.gather(grad, data))
            elif op == "transpose":
                accumulate(args[0], self.transpose(grad, data[1], data[0]))
            elif op == "matmul":
                x, weight = args
                rows, inputs, outputs = data
                accumulate(
                    x,
                    self.matmul(
                        grad,
                        self.transpose(weight, inputs, outputs),
                        rows,
                        outputs,
                        inputs,
                    ),
                )
                accumulate(
                    weight,
                    self.matmul(
                        self.transpose(x, rows, inputs), grad, inputs, rows, outputs
                    ),
                )
            elif op == "product":
                metadata, output, required = data
                values = list(args)
                values[output] = grad
                for role, x in enumerate(args):
                    if role != output:
                        accumulate(
                            x,
                            self.product(
                                values, metadata, role, required | (1 << output)
                            ),
                        )
            else:
                raise NotImplementedError(f"No native adjoint for {op!r}.")
        return tuple(results)


@lru_cache(maxsize=128)
def next_adjoint(metadata, active, seeds, num_inputs):
    """Transpose an existing derivative expression at any derivative order."""
    nodes, outputs = parse_metadata(metadata)
    program = Program(nodes)
    roots, gradients = [], []
    for root, slot, kind in outputs:
        if slot in seeds:
            roots.append(root)
            gradients.append(
                program.input(num_inputs + seeds.index(slot), kind, program.size(root))
            )
    result = program.adjoint(roots, gradients, active)
    return repr((tuple(program.nodes), result))
