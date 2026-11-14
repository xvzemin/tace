"""O3 Cartesian decomposition using shared Torch stages."""

from ...common.torch import ictd


def forward(module, features):
    return ictd.forward(module, features)


def inverse(module, features):
    return ictd.inverse(module, features)


def project(module, features, index):
    return ictd.project(module, features, index)
