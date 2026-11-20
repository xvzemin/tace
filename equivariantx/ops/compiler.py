"""Compiler passes and AOTInductor packaging."""

import hashlib
import re
from pathlib import Path
from uuid import uuid4

import torch

try:
    from torch._inductor.custom_graph_pass import CustomGraphPass
except ImportError:
    CustomGraphPass = object

__all__ = ["JointCSE", "compile_and_package"]


def compile_and_package(exported, *, package_path=None, inductor_configs=None):
    """Compile an exported graph into an AOTInductor package.

    Parameters
    ----------
    exported : torch.export.ExportedProgram
        Exported tensor program.
    package_path : str or pathlib.Path, optional
        Output ``.pt2`` path.
    inductor_configs : dict, optional
        Inductor compilation options.

    Examples
    --------
    >>> path = compile_and_package(exported, package_path="model.pt2")  # doctest: +SKIP
    """
    from torch._inductor import aoti_compile_and_package, config

    options = dict(inductor_configs or {})
    if hasattr(config.aot_inductor, "model_name_for_generated_files") and any(
        isinstance(value := node.meta.get("val"), torch.Tensor) and value.is_cuda
        for node in exported.graph.nodes
        if node.op == "placeholder"
    ):
        # Inductor caches CUDA binaries by kernel name across graphs in a process.
        # Distinguish exports even when they overwrite the same output package.
        options.setdefault(
            "aot_inductor.model_name_for_generated_files", f"eqx_{uuid4().hex}"
        )
    return aoti_compile_and_package(
        exported, package_path=package_path, inductor_configs=options
    )


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
    >>> package = compile_and_package(  # doctest: +SKIP
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
