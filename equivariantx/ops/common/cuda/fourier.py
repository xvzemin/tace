"""Warp reductions for vectorized Fourier mode projections."""

from .pointwise import launch, scalar_type


def evaluate(x, frequencies, coefficients, num_cos, extent, order, indices=None):
    channels, terms = coefficients.shape[-2:]
    output = x.new_empty((x.numel(), channels))
    if not x.numel():
        return output
    families = []
    for family, (start, stop) in enumerate(((0, num_cos), (num_cos, channels))):
        if start == stop:
            continue
        trig = "sin" if (family + order) % 2 else "cos"
        if scalar_type(x.dtype) == "float":
            trig += "f"
        sign = "-" if (order + (1 - family)) % 4 >= 2 else ""
        factors = "*w" * order
        families.append(f"""
        {{
            T values[tiles];
            #pragma unroll
            for (int t = 0; t < tiles; ++t) {{
                const int k = lane + 32*t;
                if (k < terms) {{
                    const T w = freq[{family}*terms + k];
                    values[t] = {sign}{trig}(xv*w){factors};
                }} else values[t] = T(0);
            }}
            for (int c = {start}; c < {stop}; ++c) {{
                T sum = 0;
                #pragma unroll
                for (int t=0; t < tiles; ++t) {{
                    int k = lane + 32*t;
                    if (k < terms) sum += values[t] * table[c*terms+k];
                }}
                #pragma unroll
                for (int s=16; s; s/=2) sum += __shfl_down_sync(0xffffffffu, sum, s);
                if (lane == 0) out[row*channels+c] = sum;
            }}
        }}""")
    source = f"""
    using T = {scalar_type(x.dtype)};
    extern "C" __global__ void run(const T* x, const T* freq, const T* coeff,
                                   {"const long long* indices," if indices is not None else ""} T* out, long long n) {{
        constexpr int channels={channels}, terms={terms}, tiles=(terms+31)/32;
        const int lane=threadIdx.x%32;
        for (long long row=((long long)blockIdx.x*blockDim.x+threadIdx.x)/32;
             row<n; row+=(long long)gridDim.x*blockDim.x/32) {{
            const T xv = x[row];
            {"const long long id = indices[row];" if indices is not None else ""}
            if (!(xv >= T(0) && xv < T({float(extent).hex()}))
                {f"|| id < 0 || id >= {coefficients.shape[0]}" if indices is not None else ""}) {{
                for (int c=lane; c<channels; c+=32) out[row*channels+c] = T(0);
                continue;
            }}
            const T* table = coeff {"+ id*channels*terms" if indices is not None else ""};
            {"".join(families)}
        }}
    }}"""
    launch(
        source,
        (
            x.contiguous(),
            frequencies.contiguous(),
            coefficients.contiguous(),
            *((indices.contiguous(),) if indices is not None else ()),
            output,
        ),
        (x.numel(),),
        x.numel() * 32,
        threads=128,
    )
    return output
