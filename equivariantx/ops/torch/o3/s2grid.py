"""Fixed spherical quadratures with exact linear adjoints."""

from ... import config
from ...common.torch import grid


def to_grid_reference(module, features):
    return features @ module.synthesis.T


def from_grid_reference(module, features):
    return features @ module.analysis.T


def to_grid(module, features):
    if config.TORCH_REFERENCE or not config.TORCH_S2_GRID:
        return to_grid_reference(module, features)
    return grid.linear(features, module.synthesis)


def from_grid(module, features):
    if config.TORCH_REFERENCE or not config.TORCH_S2_GRID:
        return from_grid_reference(module, features)
    return grid.linear(features, module.analysis)
