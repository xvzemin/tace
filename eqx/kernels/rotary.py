"""Head-wise rotary inner products and their recursive transposes."""

from functools import lru_cache

import torch

from .cuda import kernels, runtime


def rotary_product(query, key, phase, lmax, mmax, heads):
    empty = query.new_empty((0, heads))
    return contract(query, key, phase, empty, lmax, mmax, heads, 3)


@lru_cache(maxsize=128)
def source(dtype, lmax, mmax, channels, heads, role):
    orders, partners, signs = [], [], []
    starts, offset = [], 0
    for m in range(mmax + 1):
        n = lmax + 1 - m
        starts.append(offset)
        for real in range(1 if m == 0 else 2):
            orders.extend([m] * n)
            partners.extend(
                range(offset + (1 - real) * n, offset + (2 - real) * n)
                if m
                else range(n)
            )
            signs.extend([(-1 if real == 0 else 1) if m else 0] * n)
        offset += n * (1 if m == 0 else 2)
    angular, per_head = offset, channels // heads
    constants = "\n".join(
        f"__device__ __constant__ int {name}[]={{{','.join(map(str, values))}}};"
        for name, values in (
            ("order", orders),
            ("partner", partners),
            ("sign", signs),
            ("start", starts),
        )
    )
    if role in (0, 1):
        other = "key" if role == 0 else "query"
        cross_sign = "sign[a]" if role == 0 else "-sign[a]"
        body = f"""
        long long e=blockIdx.x; int ac=blockIdx.y*128+threadIdx.x;
        if(ac>={angular * channels}) return;
        int a=ac/{channels},c=ac%{channels},h=c/{per_head};
        long long p=((e*{mmax + 1}+order[a])*{heads}+h)*2;
        scalar value=phase[p]*{other}[e*{angular * channels}+ac];
        if(sign[a]) value+=scalar({cross_sign})*phase[p+1]*{other}[(e*{angular}+partner[a])*{channels}+c];
        result[e*{angular * channels}+ac]=value*score[e*{heads}+h];
        """
    elif role == 2:
        body = f"""
        long long e=blockIdx.x; int m=blockIdx.y/{heads},h=blockIdx.y%{heads};
        int n={lmax + 1}-m; scalar real=0,imag=0;
        for(int i=threadIdx.x;i<n*{per_head};i+=128) {{
            int a=start[m]+i/{per_head},c=h*{per_head}+i%{per_head};
            long long pos=(e*{angular}+a)*{channels}+c;
            scalar qr=query[pos],kr=key[pos]; real+=qr*kr;
            if(m) {{
                scalar qi=query[pos+n*{channels}],ki=key[pos+n*{channels}];
                real+=qi*ki; imag+=qi*kr-qr*ki;
            }}
        }}
        __shared__ scalar re[128],im[128]; re[threadIdx.x]=real; im[threadIdx.x]=imag;
        __syncthreads();
        for(int s=64;s;s>>=1) {{
            if(threadIdx.x<s) {{ re[threadIdx.x]+=re[threadIdx.x+s]; im[threadIdx.x]+=im[threadIdx.x+s]; }}
            __syncthreads();
        }}
        if(!threadIdx.x) {{
            long long p=((e*{mmax + 1}+m)*{heads}+h)*2;
            result[p]=re[0]*score[e*{heads}+h]; result[p+1]=im[0]*score[e*{heads}+h];
        }}
        """
    else:
        body = f"""
        long long e=blockIdx.x; int h=blockIdx.y; scalar sum=0;
        for(int i=threadIdx.x;i<{angular * per_head};i+=128) {{
            int a=i/{per_head},c=h*{per_head}+i%{per_head};
            long long p=((e*{mmax + 1}+order[a])*{heads}+h)*2;
            scalar k=phase[p]*key[(e*{angular}+a)*{channels}+c];
            if(sign[a]) k+=scalar(sign[a])*phase[p+1]*key[(e*{angular}+partner[a])*{channels}+c];
            sum+=query[(e*{angular}+a)*{channels}+c]*k;
        }}
        __shared__ scalar values[128]; values[threadIdx.x]=sum; __syncthreads();
        for(int s=64;s;s>>=1) {{
            if(threadIdx.x<s) values[threadIdx.x]+=values[threadIdx.x+s];
            __syncthreads();
        }}
        if(!threadIdx.x) result[e*{heads}+h]=values[0];
        """
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
    using scalar={scalar};
    {constants}
    extern "C" __global__ void run(const scalar* query,const scalar* key,
        const scalar* phase,const scalar* score,scalar* result) {{ {body} }}
    """


@torch.library.custom_op("eqx::rotary_product", mutates_args=(), device_types="cuda")
def contract(
    query: torch.Tensor,
    key: torch.Tensor,
    phase: torch.Tensor,
    score: torch.Tensor,
    lmax: int,
    mmax: int,
    heads: int,
    role: int,
) -> torch.Tensor:
    query, key, phase, score = (x.contiguous() for x in (query, key, phase, score))
    result = contract_fake(query, key, phase, score, lmax, mmax, heads, role)
    edges, angular, channels = query.shape
    if edges:
        code = source(query.dtype, lmax, mmax, channels, heads, role)
        compiled = kernels([code], query.device)
        groups = (
            (angular * channels + 127) // 128
            if role < 2
            else (mmax + 1) * heads
            if role == 2
            else heads
        )
        args = [x.data_ptr() for x in (query, key, phase, score, result)]
        runtime().launch(
            [(compiled[code], args, edges, groups, 128, 0)],
            torch.cuda.current_stream(query.device).cuda_stream,
        )
    return result


@contract.register_fake
def contract_fake(query, key, phase, score, lmax, mmax, heads, role):
    if role < 2:
        return torch.empty_like(query)
    if role == 2:
        return torch.empty_like(phase)
    return query.new_empty((query.shape[0], heads))


def setup_context(ctx, inputs, output):
    query, key, phase, score, ctx.lmax, ctx.mmax, ctx.heads, ctx.role = inputs
    ctx.save_for_backward(query, key, phase, score)


def backward(ctx, grad):
    values = list(ctx.saved_tensors)
    values[ctx.role] = grad
    gradients = [
        contract(*values, ctx.lmax, ctx.mmax, ctx.heads, role)
        if role != ctx.role and ctx.needs_input_grad[role]
        else None
        for role in range(4)
    ]
    return (*gradients, None, None, None, None)


contract.register_autograd(backward, setup_context=setup_context)
