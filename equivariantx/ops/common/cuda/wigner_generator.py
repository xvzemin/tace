"""Packed degree recurrence and infinitesimal rotation contractions."""

from functools import lru_cache
from math import sqrt

import torch
from e3nn import o3

from ....o3.rotation import so3_generators
from .runtime import kernels


@lru_cache(maxsize=48)
def coefficients(degree, generators):
    pointers, indices, values = [0], [], []
    for l in range(degree + 1):
        if generators:
            matrix = so3_generators(l).transpose(0, 1)
        elif l >= 2:
            cg = o3.wigner_3j(1, l - 1, l, dtype=torch.float64, device="cpu")
            matrix = cg.permute(2, 0, 1) * sqrt(2 * l + 1)
        else:
            pointers.extend([len(indices)] * (2 * l + 1))
            continue
        for row in matrix:
            for a, b in row.nonzero().tolist():
                indices.append((a, b))
                values.append(row[a, b].item())
            pointers.append(len(indices))
    return pointers, indices, values


@lru_cache(maxsize=128)
def source(degree, dtype, mode):
    ptr, ind, values = coefficients(degree, mode != 0)
    width = (degree + 1) * (2 * degree + 1) * (2 * degree + 3) // 3
    size = 8 if dtype == "double" else 4
    memory = (
        "__constant__"
        if 4 * len(ptr) + (4 + size) * len(values) < 65520
        else "__device__ const"
    )

    def array(type_, name, seq):
        return f"{memory} {type_} {name}[]={{" + ",".join(map(str, seq or [0])) + "};\n"

    header = (
        f"using T={dtype};\n"
        + array("int", "ptr", ptr)
        + array("short", "ax", [a for a, _ in ind])
        + array("short", "col", [b for _, b in ind])
        + array("T", "val", [f"T({x:.17g})" for x in values])
    )
    header += f"constexpr int L={degree}, W={width};\n"
    entry = 'extern "C" __global__ void run(const T* x,const T* y,T* out){\n'
    if mode == 0:
        # Keep two preceding blocks in shared memory while they fit in 48 KiB.
        # Larger blocks are already present in the packed result in global memory.
        shared_degree = min(degree, 27 if size == 8 else 38)
        return (
            header
            + entry
            + f"constexpr int S={shared_degree};\n"
            + """
        const auto e=(long long)blockIdx.x; const int t=threadIdx.x;
        const T* q=x+4*e; T* d=out+W*e;
        __shared__ T a[9], p[2][(S>0?(2*S+1)*(2*S+1):9)];
        if(t==0){
          d[0]=1;
          T w=q[0],i=q[1],j=q[2],k=q[3];
          a[0]=w*w+i*i-j*j-k*k; a[1]=2*(i*j-w*k); a[2]=2*(i*k+w*j);
          a[3]=2*(i*j+w*k); a[4]=w*w-i*i+j*j-k*k; a[5]=2*(j*k-w*i);
          a[6]=2*(i*k-w*j); a[7]=2*(j*k+w*i); a[8]=w*w-i*i-j*j+k*k;
        }
        __syncthreads();
        if(L && t<9) d[1+t]=p[1][t]=a[t];
        __syncthreads();
        int prev=1, offset=10;
        for(int l=2;l<=L;++l){
          int n=2*l+1;
          const T* previous=(L<=S || l-1<=S)?p[(l-1)&1]:d+prev;
          for(int h=t;h<n*n;h+=blockDim.x){
            int m=l*l+h/n, k=l*l+h%n; T v=0;
            for(int i=ptr[m];i<ptr[m+1];++i){
              T part=0;
              for(int j=ptr[k];j<ptr[k+1];++j)
                part+=val[j]*a[3*ax[i]+ax[j]]*previous[(n-2)*col[i]+col[j]];
              v+=val[i]*part;
            }
            d[offset+h]=v;
            if(L<=S || l<=S) p[l&1][h]=v;
          }
          __syncthreads(); prev=offset; offset+=n*n;
        }
        }"""
        )
    return (
        header
        + entry
        + f"constexpr bool reduce={int(mode == 2)};\n"
        + """
        const auto e=(long long)blockIdx.x; const int t=threadIdx.x;
        const T* d=x+W*e; const T* b=y+(reduce?W:4)*e;
        T v[4]={0,0,0,0}; int offset=0;
        for(int l=0;l<=L;++l){
          int n=2*l+1;
          for(int h=t;h<n*n;h+=blockDim.x){
            int row=l*l+h/n, c=h%n; T z=d[offset+h];
            T base=reduce?b[offset+h]:b[0];
            T sum=T(l)*z*base;
            if(reduce) v[0]+=sum;
            for(int j=ptr[row];j<ptr[row+1];++j){
              int a=ax[j]+1; T term=val[j]*d[offset+col[j]*n+c];
              if(reduce) v[a]+=base*term; else sum+=b[a]*term;
            }
            if(!reduce) out[e*W+offset+h]=sum;
          }
          offset+=n*n;
        }
        if(reduce){
          __shared__ T s[4][4]; int lane=t&31,warp=t>>5;
          for(int a=0;a<4;++a){
            for(int k=16;k;k/=2) v[a]+=__shfl_down_sync(0xffffffff,v[a],k);
            if(lane==0) s[a][warp]=v[a];
          }
          __syncthreads();
          if(t<4) out[4*e+t]=s[t][0]+s[t][1]+s[t][2]+s[t][3];
        }
        }"""
    )


def evaluate(degree, mode, a, b):
    width = (degree + 1) * (2 * degree + 1) * (2 * degree + 3) // 3
    result = a.new_empty((a.shape[0], 4 if mode == 2 else width))
    if result.numel():
        code = source(degree, "double" if a.dtype == torch.float64 else "float", mode)
        kernel = kernels([code], a.device)[code]
        a, b = a.contiguous(), b.contiguous()
        kernel.launch(
            [a.data_ptr(), b.data_ptr(), result.data_ptr()],
            a.shape[0],
            1,
            128,
            torch.cuda.current_stream(a.device).cuda_stream,
        )
    return result
