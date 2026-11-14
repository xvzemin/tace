"""Native sparse Cartesian products before STF projection."""

from ..channel_product import angular_product
from ..torch.cartesian import normalized_coefficients


def cartesian_product(x, y, l1, l2, l3):
    """Contract exact delta/epsilon entries with native multilinear adjoints."""
    return angular_product(x, y, normalized_coefficients(l1, l2, l3), 3**l3)
