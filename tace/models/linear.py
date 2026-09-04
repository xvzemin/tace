################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import torch
import torch.nn.functional as F
from e3nn import o3

from tace.utils.env import get_tace_use_matrix_weight


def _lora_scaling(module: torch.nn.Module) -> float:
    return float(module.lora_alpha) / float(module.lora_r)


def _lora_dtype_device(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    if isinstance(getattr(module, "weight", None), torch.nn.ParameterList):
        param = module.weight[0]
    else:
        param = module.weight
    return param.device, param.dtype


def _freeze_lora_base(module: torch.nn.Module, train_bias: bool) -> None:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.nn.ParameterList):
        for param in weight:
            param.requires_grad_(False)
    elif weight is not None:
        weight.requires_grad_(False)

    bias = getattr(module, "bias", None)
    if bias is not None:
        bias.requires_grad_(bool(train_bias))


def _init_lora_parameter(
    shape: tuple[int, ...],
    *,
    zero: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
    if zero:
        torch.nn.init.zeros_(param)
    else:
        fan_in = max(int(shape[-2]), 1)
        bound = 1.0 / math.sqrt(fan_in)
        torch.nn.init.uniform_(param, -bound, bound)
    return param


def _make_lora_path_parameters(
    module: torch.nn.Module,
    r: int,
    element_aware: bool,
    expert_aware: bool,
) -> tuple[torch.nn.ParameterList, torch.nn.ParameterList]:
    device, dtype = _lora_dtype_device(module)
    prefix = []
    if hasattr(module, "num_elements") and element_aware:
        prefix.append(int(module.num_elements))
    if hasattr(module, "num_experts") and expert_aware:
        prefix.append(int(module.num_experts))

    lora_A = torch.nn.ParameterList()
    lora_B = torch.nn.ParameterList()
    for path_shape in module._path_shapes:
        left_dim = int(path_shape[0])  # in
        right_dim = int(math.prod(path_shape[1:]))  # out
        lora_A.append(
            _init_lora_parameter((*prefix, left_dim, r), device=device, dtype=dtype)
        )
        lora_B.append(
            _init_lora_parameter(
                (*prefix, r, right_dim),
                zero=True,
                device=device,
                dtype=dtype,
            )
        )
    return lora_A, lora_B


def _lora_path_delta(module: torch.nn.Module, path_index: int) -> torch.Tensor:
    delta = module.lora_A[path_index] @ module.lora_B[path_index]
    return delta.reshape(
        *delta.shape[:-2], *module._path_shapes[path_index]
    ) * _lora_scaling(module)


def _flat_lora_delta(module: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for path_index in range(len(module._path_shapes)):
        delta = _lora_path_delta(module, path_index)
        path_ndim = len(module._path_shapes[path_index])
        chunks.append(delta.reshape(*delta.shape[:-path_ndim], -1))
    return torch.cat(chunks, dim=-1)


def enable_lora(
    module: torch.nn.Module,
    *,
    r: int = 4,
    alpha: float | None = None,
    element_aware: bool = True,
    expert_aware: bool = True,
    freeze_base: bool = True,
    train_bias: bool = False,
) -> torch.nn.Module:
    if r <= 0:
        return module

    module.lora_r = int(r)
    module.lora_alpha = float(alpha if alpha is not None else r)
    module.use_lora = True

    if isinstance(module, torchLinear):
        device, dtype = _lora_dtype_device(module)
        module.lora_A = _init_lora_parameter(
            (r, module.in_features),
            device=device,
            dtype=dtype,
        )
        module.lora_B = _init_lora_parameter(
            (module.out_features, r),
            zero=True,
            device=device,
            dtype=dtype,
        )
    elif isinstance(module, mlpLinear):
        device, dtype = _lora_dtype_device(module)
        module.lora_A = _init_lora_parameter(
            (module.in_dim, r),
            device=device,
            dtype=dtype,
        )
        module.lora_B = _init_lora_parameter(
            (r, module.out_dim),
            zero=True,
            device=device,
            dtype=dtype,
        )
    elif isinstance(module, (e3nnLinear, e3nnElementLinear, e3nnMoEElementLinear)):
        module.lora_element_aware = bool(element_aware)
        module.lora_expert_aware = bool(expert_aware)
        module.lora_A, module.lora_B = _make_lora_path_parameters(
            module,
            r,
            element_aware=element_aware,
            expert_aware=expert_aware,
        )
    else:
        raise TypeError(f"Unsupported LoRA module type: {type(module)}")

    if freeze_base:
        _freeze_lora_base(module, train_bias=train_bias)

    return module


def has_lora(module: torch.nn.Module) -> bool:
    return bool(getattr(module, "use_lora", False)) and hasattr(module, "lora_A")


def disable_lora(module: torch.nn.Module) -> torch.nn.Module:
    if hasattr(module, "use_lora"):
        module.use_lora = False
    return module


def merge_lora(module: torch.nn.Module, remove_lora: bool = True) -> torch.nn.Module:
    if not has_lora(module):
        return module

    with torch.no_grad():
        if isinstance(module, torchLinear):
            module.weight.add_((module.lora_B @ module.lora_A) * _lora_scaling(module))
        elif isinstance(module, mlpLinear):
            module.weight.add_((module.lora_A @ module.lora_B) * _lora_scaling(module))
        elif isinstance(module, (e3nnLinear, e3nnElementLinear, e3nnMoEElementLinear)):
            if isinstance(module.weight, torch.nn.ParameterList):
                for path_index, weight in enumerate(module.weight):
                    weight.add_(_lora_path_delta(module, path_index))
            else:
                module.weight.add_(_flat_lora_delta(module))
        else:
            raise TypeError(f"Unsupported LoRA module type: {type(module)}")

    if remove_lora:
        for name in (
            "lora_A",
            "lora_B",
            "lora_r",
            "lora_alpha",
            "lora_element_aware",
            "lora_expert_aware",
        ):
            if hasattr(module, name):
                delattr(module, name)
        module.use_lora = False

    return module


class torchLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.randn((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.alpha = 1.0 / math.sqrt(in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if has_lora(self):
            weight = weight + (self.lora_B @ self.lora_A) * _lora_scaling(self)
        return F.linear(input, weight * self.alpha, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class mlpLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        alpha: float = 1.0,
        bias: bool = False,
        std: float = math.sqrt(3),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.weight = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weight, -std, std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.alpha
        if has_lora(self):
            weight = (
                self.weight + (self.lora_A @ self.lora_B) * _lora_scaling(self)
            ) * self.alpha
        if self.bias is None:
            return torch.mm(input, weight)
        else:
            return torch.addmm(self.bias, input, weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim} bias={self.bias is not None})"


class e3nnLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
        internal_weights: bool = True,
        use_matrix_weight: bool = get_tace_use_matrix_weight(),
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.use_matrix_weight = use_matrix_weight == "1"

        self.linear = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        self.weight_numel = self.linear.weight_numel
        self._path_shapes = [tuple(ins.path_shape) for ins in self.linear.instructions]

        self._0e_muls = []
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        bias_acc = 0
        for mul, ir in self.irreps_out:
            dim = mul * ir.dim
            if ir.is_scalar():
                self._0e_muls.append(mul)
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_acc, bias_acc + dim))
                bias_acc += dim
            acc += dim

        if self.use_matrix_weight:
            if internal_weights:
                self.weight = torch.nn.ParameterList()
                for ins in self.linear.instructions:
                    self.weight.append(torch.nn.Parameter(torch.randn(ins.path_shape)))
            else:
                self.register_parameter("weight", None)
        else:
            if internal_weights:
                self.weight = torch.nn.Parameter(torch.empty(self.weight_numel))
                torch.nn.init.normal_(self.weight)
            else:
                self.register_parameter("weight", None)

        if bias and bias_acc > 0:
            self.bias = torch.nn.Parameter(torch.zeros(bias_acc))
        else:
            self.register_parameter("bias", None)
        bias_index = torch.zeros(self.irreps_out.dim, dtype=torch.long)
        bias_mask = torch.zeros(self.irreps_out.dim)
        for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
            bias_index[sl] = torch.arange(bias_sl.start, bias_sl.stop)
            bias_mask[sl] = 1.0
        self.register_buffer("_bias_index", bias_index, persistent=False)
        self.register_buffer("_bias_mask", bias_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        weight: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if self.weight is None:
            if weight is None:
                raise RuntimeError(
                    "Weights must be provided when internal_weights=False."
                )
        else:
            if weight is not None:
                raise RuntimeError(
                    "Weights must not be provided when internal_weights=True."
                )
            weight = self.weight
            if self.use_matrix_weight:
                if has_lora(self):
                    weight = torch.cat(
                        [
                            (w + _lora_path_delta(self, path_index)).view(-1)
                            for path_index, w in enumerate(weight)
                        ],
                        dim=-1,
                    )
                else:
                    weight = torch.cat([w.view(-1) for w in weight], dim=-1)
            elif has_lora(self):
                weight = weight + _flat_lora_delta(self)

        out = self.linear(x, weight)

        if self.bias is not None:
            bias = self.bias.index_select(0, self._bias_index)
            out = out + bias * self._bias_mask

        return out

    def __repr__(self):
        return repr(self.linear) + f"(bias={self.bias is not None})"


class e3nnElementLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
        num_elements: int,
        use_matrix_weight: bool = get_tace_use_matrix_weight(),
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_elements = num_elements
        self.use_matrix_weight = use_matrix_weight == "1"

        self.linear = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        weight_numel = self.linear.weight_numel
        self._path_shapes = [tuple(ins.path_shape) for ins in self.linear.instructions]

        self._0e_muls = []
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        bias_acc = 0
        for mul, ir in self.irreps_out:
            dim = mul * ir.dim
            if ir.is_scalar():
                self._0e_muls.append(mul)
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_acc, bias_acc + dim))
                bias_acc += dim
            acc += dim

        if self.use_matrix_weight:
            self.weight = torch.nn.ParameterList()
            for ins in self.linear.instructions:
                self.weight.append(
                    torch.nn.Parameter(torch.randn(num_elements, *ins.path_shape))
                )
            if bias and bias_acc > 0:
                self.bias = torch.nn.Parameter(torch.zeros(num_elements * bias_acc))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = torch.nn.Parameter(torch.empty(num_elements, weight_numel))
            torch.nn.init.normal_(self.weight)
            if bias and bias_acc > 0:
                self.bias = torch.nn.Parameter(torch.zeros(num_elements, bias_acc))
            else:
                self.register_parameter("bias", None)
        bias_index = torch.zeros(self.irreps_out.dim, dtype=torch.long)
        bias_mask = torch.zeros(self.irreps_out.dim)
        for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
            bias_index[sl] = torch.arange(bias_sl.start, bias_sl.stop)
            bias_mask[sl] = 1.0
        self.register_buffer("_bias_index", bias_index, persistent=False)
        self.register_buffer("_bias_mask", bias_mask, persistent=False)

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:

        if self.use_matrix_weight:
            if has_lora(self):
                weight = torch.cat(
                    [
                        (w + _lora_path_delta(self, path_index)).view(
                            self.num_elements, -1
                        )
                        for path_index, w in enumerate(self.weight)
                    ],
                    dim=-1,
                )
            else:
                weight = torch.cat(
                    [w.view(self.num_elements, -1) for w in self.weight], dim=-1
                )
            bias = (
                self.bias.view(self.num_elements, -1)
                if self.bias is not None
                else None
            )
        else:
            weight = self.weight
            if has_lora(self):
                weight = weight + _flat_lora_delta(self)
            bias = self.bias

        node_type = attrs.argmax(dim=-1)
        weight = weight[node_type]
        # weight = torch.einsum("bz,zi->bi", y, self.weight)
        out = self.linear(x, weight)
        if bias is not None:
            bias = torch.einsum("bz,zi->bi", attrs, bias)
            bias = bias.index_select(1, self._bias_index)
            out = out + bias * self._bias_mask
        return out

    def __repr__(self):
        return "Element" + repr(self.linear) + f"(bias={self.bias is not None})"


class e3nnMoEElementLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
        num_elements: int,
        num_experts: int,
        use_matrix_weight: bool = get_tace_use_matrix_weight(),
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_elements = num_elements
        self.num_experts = num_experts
        self.use_matrix_weight = use_matrix_weight == "1"

        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        invalid_in = [mul for mul, _ in self.irreps_in if mul % num_experts != 0]
        invalid_out = [mul for mul, _ in self.irreps_out if mul % num_experts != 0]
        if invalid_in or invalid_out:
            raise ValueError(
                "every input and output irrep multiplicity must be divisible by "
                f"num_experts={num_experts}, got invalid input multiplicities="
                f"{invalid_in} and output multiplicities={invalid_out}"
            )

        self.expert_irreps_in = o3.Irreps(
            [(mul // num_experts, ir) for mul, ir in self.irreps_in]
        )
        self.expert_irreps_out = o3.Irreps(
            [(mul // num_experts, ir) for mul, ir in self.irreps_out]
        )
        self.linear = o3.Linear(
            self.expert_irreps_in,
            self.expert_irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self._input_slices = list(self.irreps_in.slices())
        self._input_dims = [ir.dim for _, ir in self.irreps_in]
        self._input_expert_muls = [mul // self.num_experts for mul, _ in self.irreps_in]
        self._output_dims = [ir.dim for _, ir in self.irreps_out]
        self._output_expert_muls = [
            mul // self.num_experts for mul, _ in self.irreps_out
        ]
        self._path_shapes = [tuple(ins.path_shape) for ins in self.linear.instructions]
        self._path_metadata = [
            (ins.i_in, ins.i_out, float(ins.path_weight))
            for ins in self.linear.instructions
        ]

        if self.use_matrix_weight:
            self.weight = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.randn(num_elements, num_experts, *ins.path_shape)
                    )
                    for ins in self.linear.instructions
                ]
            )
        else:
            self.weight = torch.nn.Parameter(
                torch.randn(num_elements, num_experts, self.linear.weight_numel)
            )

        self._bias_out_indices = []
        bias_numel = 0
        for out_idx, (mul, ir) in enumerate(self.expert_irreps_out):
            if ir.is_scalar():
                self._bias_out_indices.append((out_idx, bias_numel, bias_numel + mul))
                bias_numel += mul
        if bias and bias_numel > 0:
            self.bias = torch.nn.Parameter(
                torch.zeros(num_elements, num_experts, bias_numel)
            )
        else:
            self.register_parameter("bias", None)

    def _path_weights(self, node_type: torch.Tensor) -> list[torch.Tensor]:
        if self.use_matrix_weight:
            if has_lora(self):
                return [
                    (weight + _lora_path_delta(self, path_index))[node_type]
                    for path_index, weight in enumerate(self.weight)
                ]
            return [weight[node_type] for weight in self.weight]

        weight = self.weight
        if has_lora(self):
            weight = weight + _flat_lora_delta(self)
        selected = weight[node_type]
        weights = []
        offset = 0
        for path_shape in self._path_shapes:
            size = math.prod(path_shape)
            weights.append(
                selected[:, :, offset : offset + size].reshape(
                    selected.shape[0], self.num_experts, *path_shape
                )
            )
            offset += size
        return weights

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        node_type = attrs.argmax(dim=-1)
        inputs = [
            x[:, tensor_slice].reshape(x.shape[0], self.num_experts, expert_mul, ir_dim)
            for tensor_slice, ir_dim, expert_mul in zip(
                self._input_slices,
                self._input_dims,
                self._input_expert_muls,
            )
        ]
        outputs = [
            x.new_zeros(x.shape[0], self.num_experts, expert_mul, ir_dim)
            for ir_dim, expert_mul in zip(self._output_dims, self._output_expert_muls)
        ]

        for (i_in, i_out, path_weight), weight in zip(
            self._path_metadata, self._path_weights(node_type)
        ):
            outputs[i_out] = outputs[i_out] + path_weight * torch.einsum(
                "bemd,bemn->bend",
                inputs[i_in],
                weight,
            )

        if self.bias is not None:
            bias = self.bias[node_type]
            for out_idx, start, end in self._bias_out_indices:
                outputs[out_idx] = outputs[out_idx] + bias[:, :, start:end].unsqueeze(
                    -1
                )

        return torch.cat(
            [output.reshape(output.shape[0], -1) for output in outputs],
            dim=-1,
        )

    def __repr__(self):
        return (
            "MoE"
            + repr(self.linear)
            + f"(bias={self.bias is not None}, num_experts={self.num_experts})"
        )


def switch_e3nn_weight_layout(
    model: torch.nn.Module, target: str = "toggle"
) -> torch.nn.Module:

    assert target in {"toggle", "flat", "matrix"}

    def is_supported_module(m: torch.nn.Module) -> bool:
        return (
            hasattr(m, "linear")
            and hasattr(m.linear, "instructions")
            and hasattr(m, "weight")
            and hasattr(m, "use_matrix_weight")
        )

    def is_element_linear(m: torch.nn.Module) -> bool:
        return hasattr(m, "num_elements")

    def weight_is_matrix_layout(m: torch.nn.Module) -> bool:
        return isinstance(m.weight, torch.nn.ParameterList)

    def set_weight(m: torch.nn.Module, new_weight):
        if hasattr(m, "weight"):
            delattr(m, "weight")
        m.weight = new_weight

    def flat_to_matrix(m: torch.nn.Module):
        if m.weight is None:
            return

        old_weight = m.weight
        requires_grad = old_weight.requires_grad

        matrix_weights = torch.nn.ParameterList()
        offset = 0

        with torch.no_grad():
            if is_element_linear(m):
                has_experts = hasattr(m, "num_experts")
                for ins in m.linear.instructions:
                    size = int(torch.tensor(ins.path_shape).prod().item())
                    if has_experts:
                        chunk = old_weight[:, :, offset : offset + size]
                        chunk = chunk.reshape(
                            m.num_elements,
                            m.num_experts,
                            *ins.path_shape,
                        )
                    else:
                        chunk = old_weight[:, offset : offset + size]
                        chunk = chunk.reshape(m.num_elements, *ins.path_shape)
                    matrix_weights.append(
                        torch.nn.Parameter(chunk.clone(), requires_grad=requires_grad)
                    )
                    offset += size

                # bias:
                # flat layout:   [num_elements, bias_dim]
                # matrix layout: [num_elements * bias_dim]
                if not has_experts and m.bias is not None and m.bias.dim() == 2:
                    m.bias = torch.nn.Parameter(
                        m.bias.reshape(-1).clone(),
                        requires_grad=m.bias.requires_grad,
                    )

            else:
                # old_weight: [weight_numel]
                for ins in m.linear.instructions:
                    size = int(torch.tensor(ins.path_shape).prod().item())
                    chunk = old_weight[offset : offset + size]
                    chunk = chunk.reshape(*ins.path_shape)
                    matrix_weights.append(
                        torch.nn.Parameter(chunk.clone(), requires_grad=requires_grad)
                    )
                    offset += size

        set_weight(m, matrix_weights)
        m.use_matrix_weight = True

    def matrix_to_flat(m: torch.nn.Module):
        if m.weight is None:
            return

        old_weights = list(m.weight)
        requires_grad = any(w.requires_grad for w in old_weights)

        with torch.no_grad():
            if is_element_linear(m):
                # 每个 w: [num_elements, *path_shape]
                has_experts = hasattr(m, "num_experts")
                if has_experts:
                    chunks = [
                        w.reshape(m.num_elements, m.num_experts, -1)
                        for w in old_weights
                    ]
                else:
                    chunks = [w.reshape(m.num_elements, -1) for w in old_weights]
                flat_weight = torch.cat(chunks, dim=-1)

                # bias:
                # matrix layout: [num_elements * bias_dim]
                # flat layout:   [num_elements, bias_dim]
                if not has_experts and m.bias is not None and m.bias.dim() == 1:
                    m.bias = torch.nn.Parameter(
                        m.bias.reshape(m.num_elements, -1).clone(),
                        requires_grad=m.bias.requires_grad,
                    )
            else:
                chunks = [w.reshape(-1) for w in old_weights]
                flat_weight = torch.cat(chunks, dim=-1)

        set_weight(
            m, torch.nn.Parameter(flat_weight.clone(), requires_grad=requires_grad)
        )
        m.use_matrix_weight = False

    def convert_module(m: torch.nn.Module):
        if not is_supported_module(m):
            return

        currently_matrix = weight_is_matrix_layout(m)

        if target == "toggle":
            to_matrix = not currently_matrix
        else:
            to_matrix = target == "matrix"

        if to_matrix and not currently_matrix:
            flat_to_matrix(m)
        elif not to_matrix and currently_matrix:
            matrix_to_flat(m)

    for module in model.modules():
        convert_module(module)

    return model
