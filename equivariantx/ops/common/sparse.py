"""CPU metadata for sparse bilinear coefficient tensors."""

import torch


def bilinear_rows(coefficients):
    """Pack all nonzero (input, multiplier, output) entries by output row.

    Parameters
    ----------
    coefficients : torch.Tensor
        CPU float64 coefficients with three axes.

    Returns
    -------
    tuple of torch.Tensor
        Input columns, multiplier columns, and padded nonzero coefficients.
    """
    rows = [
        torch.nonzero(coefficients[..., k], as_tuple=False)
        for k in range(coefficients.shape[-1])
    ]
    width = max(1, max((row.shape[0] for row in rows), default=0))
    indices = torch.zeros((len(rows), width), dtype=torch.long, device="cpu")
    columns = torch.zeros_like(indices)
    values = coefficients.new_zeros((len(rows), width))
    for k, row in enumerate(rows):
        count = row.shape[0]
        indices[k, :count], columns[k, :count] = row[:, 0], row[:, 1]
        values[k, :count] = coefficients[row[:, 0], row[:, 1], k]
    return indices, columns, values


def matrix_rows(matrix):
    """Pack nonzero matrix entries by output row, including empty rows."""
    columns = [torch.nonzero(row, as_tuple=False).flatten() for row in matrix]
    width = max(1, max((row.numel() for row in columns), default=0))
    indices = torch.zeros((len(columns), width), dtype=torch.long, device="cpu")
    values = matrix.new_zeros(indices.shape)
    for row, selected in enumerate(columns):
        indices[row, : selected.numel()] = selected
        values[row, : selected.numel()] = matrix[row, selected]
    return indices, values
