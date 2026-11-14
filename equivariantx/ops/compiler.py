"""Optional compiler passes for explicitly differentiated training graphs."""

import hashlib
import re
from pathlib import Path

import torch

try:
    from torch._inductor.custom_graph_pass import CustomGraphPass
except ImportError:
    CustomGraphPass = object

__all__ = ["JointCSE"]


def cse_function():
    """Resolve the supported private compiler APIs when the pass is requested."""
    version = re.match(r"(\d+)\.(\d+)", str(torch.__version__))
    message = (
        "JointCSE requires PyTorch >= 2.15 with Inductor CustomGraphPass, "
        f"joint_custom_pre_pass and fx_graph_cse; found {torch.__version__}."
    )
    if version is None or tuple(map(int, version.groups())) < (2, 15):
        raise RuntimeError(message)
    try:
        from torch._functorch.compile_utils import fx_graph_cse
        from torch._inductor import config
    except ImportError as error:
        raise RuntimeError(message) from error
    if (
        CustomGraphPass is object
        or not callable(fx_graph_cse)
        or not hasattr(config, "joint_custom_pre_pass")
    ):
        raise RuntimeError(message)
    return fx_graph_cse


class JointCSE(CustomGraphPass):
    """Remove repeated expressions before Inductor lowers a joint graph.

    The graph already contains the requested loss and derivatives. Parameters
    and buffers remain separate inputs. This pass changes neither precision
    nor global compiler settings.

    Notes
    -----
    This optional feature requires PyTorch >= 2.15. It uses the private
    ``torch._functorch.compile_utils.fx_graph_cse`` and Inductor
    ``CustomGraphPass`` APIs, including upstream mutation and alias rules.
    The minimum PyTorch version for importing EQX is unchanged.

    Examples
    --------
    >>> options = {"joint_custom_pre_pass": JointCSE()}
    >>> run = torch.compile(joint_graph, options=options)  # doctest: +SKIP
    >>> package = torch._inductor.aoti_compile_and_package(  # doctest: +SKIP
    ...     exported, inductor_configs=options
    ... )
    """

    def __init__(self):
        cse_function()

    def __call__(self, graph):
        module = graph.owning_module
        if module is None:
            raise ValueError("JointCSE requires a graph owned by a GraphModule.")
        module.graph = cse_function()(graph)
        module.graph.eliminate_dead_code()

    def uuid(self):
        """Identify the implementation and PyTorch build in compiler caches."""
        return str(torch.__version__), hashlib.sha256(
            Path(__file__).read_bytes()
        ).digest()
