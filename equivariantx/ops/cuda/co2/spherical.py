"""Native sparse angular recurrences visible to joint AOT compilation."""

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda.sparse_axis import evaluate
from ...torch.co2 import spherical as torch_kernel


def forward(module, features, vectors, method):
    if forward_ad_active() or (config.SPHERICAL_INLINE and torch_kernel.compiling()):
        return torch_kernel.forward(module, features, vectors, method)
    if torch_kernel.shared_matrix(features, vectors):

        def operation(module, x, v, method):
            return torch_kernel.evaluate(module, x, v, method, evaluate)

        return torch_kernel.channel_matrix(module, features, vectors, method, operation)
    return torch_kernel.evaluate(module, features, vectors, method, evaluate)
