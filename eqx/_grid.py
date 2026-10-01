"""Fixed quadrature transforms."""

import torch


class Grid(torch.nn.Module):
    """Store fixed analysis and synthesis matrices for a quadrature grid."""

    def __init__(self, constants, dtype=None, device=None):
        super().__init__()
        self.grid_shape = tuple(constants["weights"].shape)
        self.dim = constants["synthesis"].shape[-1]
        self._constants = constants
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.get_default_device() if device is None else device
        for name, value in constants.items():
            self.register_buffer(
                name, value.to(dtype=dtype, device=device, copy=True), persistent=False
            )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, value in self._constants.items():
            self._buffers[name] = value.to(self._buffers[name], copy=True)
        return self

    def forward(self, features):
        """Evaluate coefficients on the grid; equivalent to :meth:`to_grid`."""
        return self.to_grid(features)

    def to_grid(self, features):
        """Evaluate a band-limited signal.

        Parameters
        ----------
        features : torch.Tensor
            Coefficients of shape ``(..., dim)``. Leading dimensions may
            include independent batch and channel axes.

        Returns
        -------
        torch.Tensor
            Values of shape ``(..., *grid_shape)``.
        """
        if features.ndim < 1 or features.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} coefficients on the last axis.")
        values = features @ self.synthesis.reshape(-1, self.dim).T
        return values.reshape(*features.shape[:-1], *self.grid_shape)

    def from_grid(self, features):
        """Project grid values onto the retained coefficients.

        Parameters
        ----------
        features : torch.Tensor
            Values of shape ``(..., *grid_shape)``.

        Returns
        -------
        torch.Tensor
            Coefficients of shape ``(..., dim)``.
        """
        ndim = len(self.grid_shape)
        if features.ndim < ndim or features.shape[-ndim:] != self.grid_shape:
            raise ValueError(f"Expected trailing grid dimensions {self.grid_shape}.")
        return features.flatten(-ndim) @ self.analysis.reshape(self.dim, -1).T
