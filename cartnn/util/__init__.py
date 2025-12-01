from .default_type import (
    torch_get_default_tensor_type,
    torch_get_default_device,
    explicit_default_types,
)
from .torch_scatter import (
    scatter, 
    scatter_mul, 
    scatter_add, 
    scatter_max, 
    scatter_mean, 
    scatter_min, 
    scatter_sum,
)

def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


__all__ = [
    "scatter", 
    "scatter_mul", 
    "scatter_add", 
    "scatter_max", 
    "scatter_mean", 
    "scatter_min", 
    "scatter_sum",
    
    "torch_get_default_tensor_type",
    "torch_get_default_device",
    "explicit_default_types",
    "prod",
]
