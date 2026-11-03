import math
import torch


from ..irreps import check_irreps
from ..structs import TensorProductInfo, tp_infos
from .sparse_product import sparse_mul


class TensorProductUUUDummy(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        input1: torch.Tensor, 
        input2: torch.Tensor, 
        weight: torch.Tensor,
        tp_info_forward: TensorProductInfo,
        tp_info_backward1: TensorProductInfo,
        tp_info_backward2: TensorProductInfo
    ):
        
        inter = sparse_mul(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_mul(inter, weight, tp_info_forward.info_M_fwd)

        grad_weight = weight.requires_grad
        grad_input1 = input1.requires_grad
        grad_input2 = input2.requires_grad
        
        ctx.inter_shape = inter.shape
        ctx.weight_ndim = weight.ndim
        if not grad_weight:
            inter = None
        if (not grad_input1) and (not grad_input2):
            weight = None
        if not grad_input1:
            input2 = None
        if not grad_input2:
            input1 = None
        ctx.save_for_backward(input1, input2, weight, inter)
        ctx.tp_info = (tp_info_forward, tp_info_backward1, tp_info_backward2)
        return ret, inter

    @staticmethod
    def backward(ctx, grad_output, grad_inter):
        grad = grad_output
        input1, input2, weight, inter = ctx.saved_tensors
        tp_info_forward, tp_info_backward1, tp_info_backward2 = ctx.tp_info

        # ``inter`` is an auxiliary output used only when a higher-order
        # derivative propagates through the weight gradient.  In the usual
        # first-derivative path it is not consumed, so ``grad_inter`` is None
        # and its contribution is exactly zero.  Avoid materializing that
        # potentially large zero tensor and the two sparse products fed by it.
        if grad_inter is not None:
            grad_inter = torch.broadcast_to(grad_inter, ctx.inter_shape)
        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_uuu(
                input2,
                grad,
                weight,
                tp_info_backward1,
                tp_info_backward2,
                tp_info_forward,
            )
            if grad_inter is not None:
                grad1 = grad1 + sparse_mul(
                    input2,
                    grad_inter,
                    tp_info_forward.info_Mij_bwd1,
                    tp_info_forward.info_Mij_bwd2,
                    tp_info_forward.info_Mij_fwd,
                )
        else:
            grad1 = None

        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_uuu(
                grad,
                input1,
                weight,
                tp_info_backward2,
                tp_info_forward,
                tp_info_backward1,
            )
            if grad_inter is not None:
                grad2 = grad2 + sparse_mul(
                    grad_inter,
                    input1,
                    tp_info_forward.info_Mij_bwd2,
                    tp_info_forward.info_Mij_fwd,
                    tp_info_forward.info_Mij_bwd1,
                )
        else:
            grad2 = None

        if ctx.needs_input_grad[2]:
            grad_W = sparse_mul(grad_output, inter,
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 2)
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None


def tensor_product_uuu(
        input1: torch.Tensor, 
        input2: torch.Tensor, 
        weight: torch.Tensor,
        tp_info_forward: TensorProductInfo,
        tp_info_backward1: TensorProductInfo = None,
        tp_info_backward2: TensorProductInfo = None
    ):
    ret, _ =  TensorProductUUUDummy.apply(input1, input2, weight,
                                       tp_info_forward,
                                       tp_info_backward1,
                                       tp_info_backward2)
    return ret


class TensorProduct(torch.nn.Module):

    tp_info_forward: TensorProductInfo
    tp_info_backward1: TensorProductInfo
    tp_info_backward2: TensorProductInfo

    def __init__(self, 
                 irreps_in1,
                 irreps_in2, 
                 irreps_out, 
                 channels_in1=None,
                 channels_in2=None,
                 channels_out=None,
                 internal_weights=True,
                 path_norm=True,
                 trainable: bool = True,
                 path=None):

        super().__init__()

        self.irreps_in1 = check_irreps(irreps_in1)
        self.irreps_in2 = check_irreps(irreps_in2)
        self.irreps_out = check_irreps(irreps_out)
        self.irreps_in1_dim = self.irreps_in1.dim
        self.irreps_in2_dim = self.irreps_in2.dim
        self.irreps_out_dim = self.irreps_out.dim
        self.trainable = trainable

        assert not internal_weights or (
            channels_in1 is not None and
            channels_in2 is not None and
            channels_out is not None
        )
        self.channels_in1 = channels_in1
        self.channels_in2 = channels_in2
        self.channels_out = channels_out
        self.path_norm = path_norm

        (self.tp_info_forward, 
          self.tp_info_backward1,
          self.tp_info_backward2,
          self.num_paths) = tp_infos(
              self.irreps_out,
              self.irreps_in1,
              self.irreps_in2, path=path, path_norm=path_norm,
              channel_norm=False, channel_scale=1**(-0.5)
          )

        self.weight_shape = (self.num_paths, self.channels_out)
        self.weight_numel = math.prod(self.weight_shape)

        self.internal_weights = internal_weights
        if internal_weights:
            if self.trainable:
                self.weight = torch.nn.Parameter(torch.empty(*self.weight_shape))
                a =  3 ** 0.5
                torch.nn.init.uniform_(self.weight, -a, a)
            else:
                self.register_buffer('weight', torch.ones(*self.weight_shape), persistent=False)
        else:
            self.weight = None

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        assert input1.shape[-2] == self.irreps_in1_dim, f"Input1 spherical dim mismatch: expected {self.irreps_in1_dim}, got {input1.shape[-2]}"
        assert input2.shape[-2] == self.irreps_in2_dim, f"Input2 spherical dim mismatch: expected {self.irreps_in2_dim}, got {input2.shape[-2]}"
        assert input1.shape[-1] == input2.shape[-1] == self.channels_in1, f"Input1 channel dim mismatch: expected {self.channels_in1}, got {input1.shape[-1]}"

        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        args = (input1, input2, weight,
                self.tp_info_forward,
                self.tp_info_backward1,
                self.tp_info_backward2)

        return tensor_product_uuu(*args)

    def _apply(self, *args, **kwargs):
        tp = super()._apply(*args, **kwargs)
        tp.tp_info_forward = self.tp_info_forward._apply(*args, **kwargs)
        tp.tp_info_backward1 = self.tp_info_backward1._apply(*args, **kwargs)
        tp.tp_info_backward2 = self.tp_info_backward2._apply(*args, **kwargs)
        return tp

    def __repr__(self):

        def to_beautiful(irreps, channel):
            return "+".join(f"{channel}x{irrep}" for irrep in irreps.short_repr().split('+'))

        return (
            f"{self.__class__.__name__}("
            f"{to_beautiful(self.irreps_in1.simplified(), self.channels_in1)} x "
            f"{to_beautiful(self.irreps_in2.simplified(), self.channels_in2)} -> "
            f"{to_beautiful(self.irreps_out.simplified(), self.channels_out)})"
        )
