"""Versioned post-autodiff extension boundary, with no extensions enabled.

Future compensated GEMMs can be introduced here after an explicit policy
change and mixed-derivative tests. A graph pass receives live operands; it must
not capture parameter values. The current contract forbids reduced arithmetic,
including hidden casts inside a custom operator.
"""

import hashlib
import os


GRAPH_EXTENSIONS = {}
LOCKED = False


def register_graph_extension(name, version, transform):
    """Register a precision-preserving graph transformation before compilation.

    Parameters
    ----------
    name, version : str
        Stable identity. Change the version whenever generated code changes.
    transform : callable
        Mutates the already differentiated FX graph. It must preserve FP32,
        FP64 or TF32 arithmetic, including inside opaque custom operators.

    Examples
    --------
    >>> # register_graph_extension("my_fp32_gemm", "v1", replace_gemm)
    """
    if LOCKED:
        raise RuntimeError("Register extensions before creating any compiled program")
    if not name or not version or not callable(transform):
        raise ValueError("Supply a nonempty name/version and a callable graph pass")
    if name in GRAPH_EXTENSIONS:
        raise ValueError(f"Extension already registered: {name}")
    GRAPH_EXTENSIONS[name] = (version, transform)


def extension_identity():
    """Freeze the registry and include extension versions in compiler keys."""
    global LOCKED
    LOCKED = True
    identity = tuple((name, value[0]) for name, value in GRAPH_EXTENSIONS.items())
    result = hashlib.sha256(repr(identity).encode()).hexdigest()
    # TACE's persisted Joint artifact lookup precedes Inductor's pass UUID.
    # Its runtime key includes EQX_* environment values as well.
    os.environ["EQX_XL_EXTENSION_IDENTITY"] = result
    return result


def apply_graph_extensions(graph):
    """Apply registered passes in registration order before native scheduling."""
    extension_identity()
    for _, transform in GRAPH_EXTENSIONS.values():
        transform(graph)
