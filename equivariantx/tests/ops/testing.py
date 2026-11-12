"""Independent values for compiled and eager multilinear transforms."""

import torch
from torch.fx.experimental.proxy_tensor import make_fx


def assert_functional_derivatives(function, reference, values, compiler=None):
    """Compare JVP, JVP of a gradient and a batch with distinct index maps."""
    primals, types = values[:-1], values[-1]
    tangents = tuple(torch.randn_like(value) for value in primals)
    arguments = tuple(range(len(primals)))

    def transformed(function, *values):
        def continuous(*primals):
            return function(*primals, values[-1])

        gradient = torch.func.grad(
            lambda *primals: continuous(*primals).sin().sum(), argnums=arguments
        )
        return (
            torch.func.jvp(continuous, values[:-1], tangents),
            torch.func.jvp(gradient, values[:-1], tangents),
        )

    expected = transformed(reference, *values)
    torch.compiler.reset()
    run = lambda *values: transformed(function, *values)
    if compiler:
        run = torch.compile(run, backend=compiler, fullgraph=True)
    torch.testing.assert_close(run(*values), expected, atol=2e-9, rtol=2e-9)

    def nested(function, *values):
        continuous = lambda *inputs: function(*inputs, values[-1])
        return torch.func.jvp(
            lambda *inputs: torch.func.jvp(continuous, inputs, tangents)[1],
            values[:-1],
            tangents,
        )

    expected = nested(reference, *values)
    run = lambda *values: nested(function, *values)
    torch.testing.assert_close(run(*values), expected, atol=2e-9, rtol=2e-9)
    if compiler:
        graph = make_fx(run)(*values)
        graph.graph.eliminate_dead_code()
        graph.recompile()
        torch.testing.assert_close(
            torch.compile(graph, backend=compiler, fullgraph=True)(*values),
            expected,
            atol=2e-9,
            rtol=2e-9,
        )
    batched = tuple(torch.stack((value, value * 0.7)) for value in primals)
    batched = (*batched, torch.stack((types, types.roll(1))))
    torch.testing.assert_close(
        torch.vmap(function)(*batched),
        torch.vmap(reference)(*batched),
        atol=2e-10,
        rtol=2e-10,
    )
