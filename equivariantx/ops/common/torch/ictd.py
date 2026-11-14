"""Complete and single-path irreducible Cartesian transforms."""

import torch

from ... import config
from ..ictd import operands, path_basis
from . import sparse_chain


def reference(module, features, reverse=False):
    matrix = module.change_of_basis
    return features @ (matrix.T if reverse else matrix)


def project_reference(module, features, index):
    matrix = module.path_matrix(index)
    return (features @ matrix) @ matrix.T


def use_dense(module, features):
    return module.dim <= config.ICTD_DENSE_MAX_WIDTH and (
        not features.is_cuda or features.dtype == torch.float32
    )


def dense_project(module, features, index, operation=None):
    matrix = path_basis(module, index)
    if operation is None:
        return (features @ matrix) @ matrix.T
    return operation(operation(features, matrix.T), matrix)


def transform(module, features, reverse=False, operation=sparse_chain.evaluate):
    if not hasattr(module, "stage_names"):
        return reference(module, features, reverse)
    return operation(features, module.lefts, reverse, *operands(module))


def projection(module, features, index, operation=sparse_chain.evaluate):
    if not hasattr(module, "transition_names"):
        return project_reference(module, features, index)
    tables = operands(module, index)
    value = operation(features, module.lefts, False, *tables)
    return operation(value, module.lefts, True, *tables)


def forward(module, features):
    if config.TORCH_REFERENCE or not config.TORCH_ICTD or use_dense(module, features):
        return reference(module, features)
    operation = (
        sparse_chain.evaluate if config.TORCH_ICTD_ADJOINT else sparse_chain.primal
    )
    return transform(module, features, False, operation)


def inverse(module, features):
    if config.TORCH_REFERENCE or not config.TORCH_ICTD or use_dense(module, features):
        return reference(module, features, True)
    operation = (
        sparse_chain.evaluate if config.TORCH_ICTD_ADJOINT else sparse_chain.primal
    )
    return transform(module, features, True, operation)


def project(module, features, index):
    if config.TORCH_REFERENCE or not config.TORCH_ICTD:
        return project_reference(module, features, index)
    if use_dense(module, features):
        return dense_project(module, features, index)
    operation = (
        sparse_chain.evaluate if config.TORCH_ICTD_ADJOINT else sparse_chain.primal
    )
    return projection(module, features, index, operation)
