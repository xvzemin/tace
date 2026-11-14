"""Native sparse real O(2) products and exact recursive adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from equivariantx.o2.irreps import Irrep

from ... import config
from ...common.cuda.sparse_bilinear import evaluate as operation
from ...torch.o2.clebsch_gordan import evaluate as torch_evaluate
from ...torch.o2.clebsch_gordan import sparse


def evaluate(input1, ir1, input2, ir2, irreps_out, elementwise=False):
    if config.O2_COUPLING_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        return torch_evaluate(input1, ir1, input2, ir2, irreps_out, elementwise)
    return native(
        input1,
        input2,
        (ir1.m, ir1.p),
        (ir2.m, ir2.p),
        tuple((ir.m, ir.p) for ir in irreps_out),
        elementwise,
    )


@torch.compiler.allow_in_graph
def native(x, y, left, right, outputs, elementwise):
    """Expose tensor operations while specializing fixed representation metadata."""
    ir1, ir2 = Irrep(*(int(v) for v in left)), Irrep(*(int(v) for v in right))
    irreps_out = tuple(Irrep(*(int(v) for v in entry)) for entry in outputs)
    return sparse(x, ir1, y, ir2, irreps_out, elementwise, operation)
