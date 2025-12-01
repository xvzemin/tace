###########################################################################################
# Authors: Zemin Xu
# This program is distributed under the MIT License
###########################################################################################

import string
from typing import List
import torch

from cartnn import o3
from cartnn.util.jit import compile_mode



@compile_mode("script")
class ChangeOfBasis(torch.nn.Module):
    def __init__(
        self,
        irreps_in: str | o3.Irreps,
        irreps_out: str | o3.Irreps,
        from_cart: bool = True,
        dim=-1,
        ndim=2,
    ) -> None:
        super().__init__()

        assert irreps_in is not None
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        def dim(ir: o3.Irrep):
            return ir.cdim if from_cart else ir.sdikm

        def Q(ir: o3.Irrep, cart: bool):
            return ir.C if cart else ir.S

        self.in_mul_list = []
        self.in_dim_list = []
        self.out_mul_list = []
        self.out_dim_list = []

        for mul, ir in irreps_in:
            self.in_mul_list.append(mul)
            self.in_dim_list.append(dim(ir))
            self.register_buffer(f'Qin{ir}', Q(ir, from_cart))
        
        for mul, ir in irreps_out:
            self.out_mul_list.append(mul)
            self.out_dim_list.append(dim(ir.l))
            self.register_buffer(f'Qout{ir}', Q(ir, (not from_cart)))

        self.in_slices = irreps_in.slices(from_cart)
        self.out_slices = irreps_out.slices(from_cart)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        strs = list(string.ascii_letters)
        in1 = strs[:ndim]
        in2 = in1[dim] + strs[ndim]
        out = in1.copy()
        out[dim] = strs[ndim]
        self.einsum_str = "".join(in1) + "," + in2 + "->" + "".join(out)

    def Q(self, ir: o3.Irrep, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return dict(self.named_buffers())[f"Qout{ir}"]
        else:
            return dict(self.named_buffers())[f"Qin{ir}"]
              
    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    
        if inverse:
            slices = self.out_slices
            irreps = self.irreps_out
            muls = self.out_mul_list
            dims = self.out_dim_list
        else:
            slices = self.in_slices
            irreps = self.irreps_in
            muls = self.in_mul_list
            dims = self.in_dim_list

        B = x.size(0)
        xs: List[torch.Tensor] = []
        for s, mul, dim, ir in zip(slices, muls, dims, irreps):
            xs.append(
                torch.einsum(
                    self.einsum_str, 
                    x[:, s].view(B, mul, dim), self.Q(ir, inverse)
                ).view(B, mul * dim)
            )
        return torch.cat(xs, dim=-1)
    

