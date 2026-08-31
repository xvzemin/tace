import logging

try:
    from torch_scatter import segment_csr
except (ImportError, OSError, RuntimeError):
    segment_csr = None
    logging.warning(
        "torch-scatter is unavailable; EQT will use the native PyTorch "
        "scatter fallback. \n Install a torch-scatter build matching the current "
        "PyTorch and CUDA versions to accelerate large batches.",
    )

from ._tp_uuu import e3nnEqtTensorProduct

__all__ = [
    "e3nnEqtTensorProduct",
]
