"""Cartesian projection and basis conversion for PyTorch contractions."""

from collections import defaultdict

from ....common.torch.basis import project_features
from ...co3.basis import change_basis
from ...co3.basis import project as project_blocks


def project(module, features):
    """Project flattened Cartesian features with PyTorch operations."""
    return project_features(module, features, project_blocks)


def project_columns(projection, columns, starts):
    """Apply the fixed Cartesian STF projector to coefficients before node evaluation."""
    for degree, entries in projection.groups:
        matrix = projection.projection_matrix(degree)
        matrix = matrix if degree == 2 else matrix @ matrix.T
        rows = matrix.tolist()
        for index, _ in entries:
            start, stop = starts[index : index + 2]
            values = columns[start:stop]
            transformed = []
            for row in rows:
                column = defaultdict(float)
                for scale, source in zip(row, values):
                    if scale:
                        for term, value in source.items():
                            column[term] += scale * value
                transformed.append(column)
            columns[start:stop] = transformed
