"""Element-dependent channel maps in spherical O(3) storage."""

from ..nn.element_linear import ElementLinear as ElementLinearBase


class ElementLinear(ElementLinearBase):
    """Apply element-dependent Linear paths in ``mul_ir`` layout.

    Parameters
    ----------
    linear : e3nn.o3.Linear or co3.Linear
        Unbiased map defining irreps, paths and normalization. Its parameters
        are not copied; weights are supplied to :meth:`forward`.
    backend : {"torch", "cuda"}, optional
        Execution backend. Defaults to ``"torch"``. CPU tensors use PyTorch.
    element_mode : {"single", "pair"}, optional
        Independent weights for each element, or each ordered element pair.
        Defaults to ``"single"``.

    Examples
    --------
    >>> import torch
    >>> from e3nn import o3
    >>> linear = o3.Linear("2x0e + 2x1o", "3x0e + 2x1o")
    >>> layer = ElementLinear(linear)
    >>> x = torch.randn(5, linear.irreps_in.dim)
    >>> weight = torch.randn(3, linear.weight_numel)
    >>> layer(x, weight, torch.tensor([0, 1, 2, 0, 1])).shape
    torch.Size([5, 9])
    """

    def __init__(self, linear, backend="torch", *, element_mode="single"):
        super().__init__(linear, backend, element_mode, "mul_ir")


class MoEElementLinear(ElementLinear):
    """Apply independent element-dependent maps to expert channels.

    Parameters
    ----------
    linear : e3nn.o3.Linear
        Unbiased map defining one expert's paths and normalization.
    num_experts : int
        Number of experts. Each irrep stores ``(experts, channels, angular)``.
    backend : {"torch", "cuda"}, optional
        Execution backend. Defaults to ``"torch"``. CPU tensors use PyTorch.
    element_mode : {"single", "pair"}, optional
        Single or two-endpoint element dependence. Defaults to ``"single"``.
    """

    def __init__(self, linear, num_experts, backend="torch", *, element_mode="single"):
        super().__init__(linear, backend, element_mode=element_mode)
        if not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError("num_experts must be a positive integer.")
        self.num_experts = num_experts
        self.input_dim *= num_experts
        self.output_dim *= num_experts
        self.irreps_in = type(linear.irreps_in)(
            [(mul * num_experts, ir) for mul, ir in linear.irreps_in]
        )
        self.irreps_out = type(linear.irreps_out)(
            [(mul * num_experts, ir) for mul, ir in linear.irreps_out]
        )
        self.paths = [
            (sx * num_experts, sw, sy * num_experts, ci, co, dim, scale)
            for sx, sw, sy, ci, co, dim, scale in self.paths
        ]
        self.set_metadata()
