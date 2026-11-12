"""Register native operators without importing the optional TACE adapter."""

import importlib


def register_runtime_ops():
    """Register operators before loading a cached graph without FX tracing.

    Examples
    --------
    >>> from equivariantx.models.tace.tace_oam_xl_v2 import register_runtime_ops
    >>> register_runtime_ops()
    """
    for name in (
        "edge", "packed_product", "product", "linear", "compressed", "parallel",
        "pipeline", "lt_matrix", "packed_fragments", "ordered_reductions",
        "precise_matrix", "tabulated",
    ):
        importlib.import_module(f"{__package__}.{name}")
