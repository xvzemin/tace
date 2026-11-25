"""Tile the five frozen distance heads as a single batched matrix program."""

from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.pointwise import launch


@lru_cache(maxsize=4)
def kernel(first, eps):
    bias = (
        "float av0=bias[l*128+ch],av1=bias[l*128+ch+32];"
        if first
        else "float av0=bias[l*64+ch],av1=bias[l*64+ch+32];"
    )
    ind = (
        "I xi=i*640+l*128+ch, xd=xi+rows*640; I p=l*128+ch;"
        if first
        else "I xi=(l*rows*2+i)*64+ch,xd=xi+rows*64;I p=l*64+ch;"
    )
    if first:
        src = "((x[xi]+av0)+dst[(l*nt+t[i])*128+ch])+src[(l*nt+s[i])*128+ch]"
        src2 = (
            src.replace("xi]", "xi+32]").replace("av0", "av1").replace("ch]", "ch+32]")
        )
    else:
        src, src2 = "x[xi]+av0", "x[xi+32]+av1"
    norm = f"""
    {ind} {bias}
    float a={src},b={src2};
    float av=sum(a+b)/64.f;a-=av;b-=av;
    float inv=rsqrtf(sum(a*a+b*b)/64.f+float({float(eps).hex()}));a*=inv;b*=inv;
    float u=x[xd],v=x[xd+32],mean=sum(u+v)/64.f,cov=sum(u*a+v*b)/64.f;
    float za=a*gain[p]+shift[p], zb=b*gain[p+32]+shift[p+32];
    float sa=1.f/(1.f+expf(-za)),sb=1.f/(1.f+expf(-zb));
    float ha=za*sa,hb=zb*sb;
    float da=(u-mean-a*cov)*inv*gain[p]*sa*(1.f+za*(1.f-sa));
    float db=(v-mean-b*cov)*inv*gain[p+32]*sb*(1.f+zb*(1.f-sb));
    """
    if first:
        ids = (
            "I block=warp/2,i=block%rows,l=block/rows;int half=warp%2,ch=lane+half*64;"
        )
        end = """
        if(half){
          float raw=sum(ha*dw[l*64+lane]+hb*dw[l*64+lane+32])+dbias[l];
          float dot=sum(da*dw[l*64+lane]+db*dw[l*64+lane+32]);
          if(!lane){float d=tanhf(raw*raw); D[l*total+offset+i]=d; DD[l*total+offset+i]=dot*(2.f*raw)*(1.f-d*d);}
        }else{
          I k=(l*2*rows+i)*64+lane;
          stage[k]=ha;stage[k+32]=hb;stage[k+rows*64]=da;stage[k+rows*64+32]=db;
        }"""
    else:
        ids = "I i=warp%rows,l=warp/rows;int ch=lane;"
        end = """I k=(l*total+offset+i)*64+ch; H[k]=ha;H[k+32]=hb; DH[k]=da;DH[k+32]=db;"""
    return (
        """
    using I=long long;
    __device__ float sum(float a){
      #pragma unroll
      for(int d=16;d;d/=2)a+=__shfl_xor_sync(0xffffffff,a,d);
      return a;
    }
    extern "C" __global__ void run(const float* x,const I* s,const I* t,const float* src,
      const float* dst,const float* bias,const float* gain,const float* shift,const float* dw,const float* dbias,
      float* stage,float* H,float* DH,float* D,float* DD,I rows,I total,I offset,I nt){
      int lane=threadIdx.x%32;
      for(I warp=(I(blockIdx.x)*blockDim.x+threadIdx.x)/32;warp<rows*"""
        + str(10 if first else 5)
        + """;
          warp+=I(gridDim.x)*blockDim.x/32){"""
        + ids
        + norm
        + end
        + "}}"
    )


@torch.library.custom_op(
    "eqx_tace_7m::batch_radial", mutates_args=(), device_types="cuda"
)
def batch_radial(
    f: torch.Tensor,
    df: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    p: list[torch.Tensor],
    eps1: float,
    eps2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w, src, dst, bias, g1, b1, w2, b2, g2, s2, dw, db = p
    n = len(f)
    h, dh = (f.new_empty((5, n, 64)) for _ in range(2))
    d, dd = (f.new_empty((5, n, 1)) for _ in range(2))
    for lo in range(0, n, 16384):
        sl = slice(lo, lo + 16384)
        r = len(f[sl])
        pre = torch.cat((f[sl], df[sl]), 0) @ w
        buf = f.new_empty((5, 2 * r, 64))
        outputs = (buf, h, dh, d, dd)
        extra = (r, n, lo, src.shape[1])
        launch(
            kernel(True, eps1),
            (pre, s[sl], t[sl], src, dst, bias, g1, b1, dw, db, *outputs),
            extra,
            r * 320,
        )
        mid = torch.bmm(buf, w2)
        launch(
            kernel(False, eps2),
            (mid, s[sl], t[sl], src, dst, b2, g2, s2, dw, db, *outputs),
            extra,
            r * 160,
        )
    return h, d, dh, dd


@batch_radial.register_fake
def fake_batch(f, df, s, t, p, eps1, eps2):
    return tuple(f.new_empty((5, f.size(0), k)) for k in (64, 1, 64, 1))
