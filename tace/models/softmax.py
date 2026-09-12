################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
Copy from
https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/models/equiformer_v3/softmax.py
https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/utils/softmax.html
"""

import torch
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tace.utils.torch_scatter import scatter_sum


class SoftCap(torch.nn.Module):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def forward(self, inputs):
        outputs = inputs / self.cap
        outputs = torch.nn.functional.tanh(outputs)
        outputs = outputs * self.cap
        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}(cap={self.cap})"


# class GraphSoftmax(torch.nn.Module):
#     def __init__(self, eps=1e-16, softcap=None):
#         super().__init__()
#         self.eps = eps
#         self.softcap = SoftCap(cap=softcap) if softcap is not None else torch.nn.Identity()

#     def forward(
#         self,
#         src,
#         index=None,
#         ptr=None,
#         num_nodes=None,
#         dim=0,
#         exp_rescale=None
#     ):
#         src = self.softcap(src)
#         if ptr is not None:
#             dim = dim + src.dim() if dim < 0 else dim
#             size = ([1] * dim) + [-1]
#             count = ptr[1:] - ptr[:-1]
#             ptr = ptr.view(size)
#             src_max = segment(src.detach(), ptr, reduce='max')
#             src_max = src_max.repeat_interleave(count, dim=dim)
#             out = (src - src_max).exp()
#             if exp_rescale is not None:
#                 out = out * exp_rescale
#             out_sum = segment(out, ptr, reduce='sum') + self.eps
#             out_sum = out_sum.repeat_interleave(count, dim=dim)
#         elif index is not None:
#             N = maybe_num_nodes(index, num_nodes)
#             src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
#             out = src - src_max.index_select(dim, index)
#             out = out.exp()
#             if exp_rescale is not None:
#                 out = out * exp_rescale
#             out_sum = scatter_sum(out, index, dim, dim_size=N) + self.eps
#             out_sum = out_sum.index_select(dim, index)
#         else:
#             raise NotImplementedError

#         out = out / out_sum

#         return out


#     def extra_repr(self):
#         return 'eps={}'.format(self.eps)


class GraphSoftmax(torch.nn.Module):
    def __init__(self, eps=1e-16):
        super().__init__()
        self.eps = eps

    def forward(
        self, src, index=None, ptr=None, num_nodes=None, dim=0, exp_rescale=None
    ):
        if ptr is not None:
            dim = dim + src.dim() if dim < 0 else dim
            size = ([1] * dim) + [-1]
            count = ptr[1:] - ptr[:-1]
            ptr = ptr.view(size)
            src_max = segment(src.detach(), ptr, reduce="max")
            src_max = src_max.repeat_interleave(count, dim=dim)
            out = (src - src_max).exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            index = torch.arange(count.numel(), device=ptr.device).repeat_interleave(
                count, output_size=out.size(dim)
            )
            out_sum = scatter_sum(out, index, dim, dim_size=count.numel()) + self.eps
            out_sum = out_sum.repeat_interleave(count, dim=dim)
        elif index is not None:
            N = maybe_num_nodes(index, num_nodes)
            src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
            out = src - src_max.index_select(dim, index)
            out = out.exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out_sum = scatter_sum(out, index, dim, dim_size=N) + self.eps
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError

        out = out / out_sum

        return out
