// Each subset stores one mixed directional derivative of the Cartesian rotation.
using I = long long;
constexpr int Masks = 1 << Directions;
constexpr int Tile = Group * Width;

__device__ void seed(T* state, const T* rows, int used) {
    for (int i = threadIdx.x; i < Masks * used; i += blockDim.x) {
        int mask = i / used, col = i % used;
        state[mask * Tile + col] = mask ? T(0) : rows[col];
    }
}

__device__ void step(const T* x, T* y, const T* maps, int axis, int stride, int used) {
    for (int i = threadIdx.x; i < Masks * used; i += blockDim.x) {
        int mask = i / used, col = i % used, o = (col / stride) % 3;
        int base = col - o * stride;
        T v = 0;
        if (__popc(unsigned(mask)) <= axis + 1) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                v += x[mask * Tile + base + j * stride] * maps[3 * j + o];
                #pragma unroll
                for (int d = 0; d < Directions; ++d)
                    if (mask & (1 << d))
                        v += x[(mask ^ (1 << d)) * Tile + base + j * stride]
                             * maps[9 * (d + 1) + 3 * j + o];
            }
        }
        y[mask * Tile + col] = v;
    }
}

extern "C" __global__ void run(const T* rows, const T* rotation, DIRECTION_ARGS
                              GRADIENT_ARG T* output, I frames) {
    extern __shared__ __align__(16) unsigned char storage[];
    T* state = reinterpret_cast<T*>(storage);
    __shared__ T maps[(Directions + 1) * 9];
    const T* matrices[] = {rotation DIRECTION_NAMES};
    constexpr int Groups = (Rows + Group - 1) / Group;
    #if TRANSPOSE
    __shared__ T warps[Threads / 32][9];
    constexpr int Parts = 1;
    #else
    constexpr int Parts = Groups;
    #endif
    for (I task = blockIdx.x; task < frames * Parts; task += gridDim.x) {
        I f = task / Parts;
        for (int i = threadIdx.x; i < (Directions + 1) * 9; i += blockDim.x)
            maps[i] = matrices[i / 9][f * 9 + i % 9];
        __syncthreads();
        const T* bank = rows + (Batched ? f * Rows * Width : 0);
        #if TRANSPOSE
        T partial[9] = {};
        for (int group = 0; group < Groups; ++group) {
            int first = group * Group, used = min(Group, Rows - first) * Width;
            seed(state, bank + first * Width, used);
            __syncthreads();
            int stride = 1;
            for (int a = 0; a < Rank - 1; ++a, stride *= 3) {
                step(state + a * Masks * Tile, state + (a + 1) * Masks * Tile,
                     maps, a, stride, used);
                __syncthreads();
            }
            T* g = state + Rank * Masks * Tile;
            T* h = g + Masks * Tile;
            for (int i = threadIdx.x; i < Masks * used; i += blockDim.x) {
                int mask = i / used, col = i % used;
                g[mask * Tile + col] = mask == Masks - 1 ?
                    gradient[(f * Rows + first) * Width + col] : T(0);
            }
            __syncthreads();
            for (int a = Rank - 1; a >= 0; --a, stride /= 3) {
                const T* x = state + a * Masks * Tile;
                for (int i = threadIdx.x; i < Masks * used; i += blockDim.x) {
                    int mask = i / used, col = i % used;
                    int o = (col / stride) % 3, base = col - o * stride;
                    T v = g[mask * Tile + col];
                    if (__popc(unsigned(mask)) <= a) {
                        #pragma unroll
                        for (int j = 0; j < 3; ++j)
                            partial[3 * j + o] += v * x[mask * Tile + base + j * stride];
                    }
                    if (a) {
                        T w = 0;
                        #pragma unroll
                        for (int j = 0; j < 3; ++j) {
                            w += maps[3 * o + j] * g[mask * Tile + base + j * stride];
                            #pragma unroll
                            for (int d = 0; d < Directions; ++d)
                                if (!(mask & (1 << d)))
                                    w += maps[(d + 1) * 9 + 3 * o + j]
                                         * g[(mask | (1 << d)) * Tile + base + j * stride];
                        }
                        h[mask * Tile + col] = w;
                    }
                }
                __syncthreads();
                T* swap = h; h = g; g = swap;
            }
        }
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            T v = partial[t];
            for (int delta = 16; delta; delta /= 2)
                v += __shfl_down_sync(0xffffffff, v, delta);
            if (!(threadIdx.x % 32)) warps[threadIdx.x / 32][t] = v;
        }
        __syncthreads();
        if (threadIdx.x < 9) {
            T v = 0;
            for (int w = 0; w < Threads / 32; ++w) v += warps[w][threadIdx.x];
            output[f * 9 + threadIdx.x] = v;
        }
        #else
        int first = (task % Groups) * Group, used = min(Group, Rows - first) * Width;
        T* x = state;
        T* y = x + Masks * Tile;
        seed(x, bank + first * Width, used);
        __syncthreads();
        for (int a = 0, stride = 1; a < Rank; ++a, stride *= 3) {
            step(x, y, maps, a, stride, used);
            __syncthreads();
            T* swap = x; x = y; y = swap;
        }
        for (int col = threadIdx.x; col < used; col += blockDim.x)
            output[(f * Rows + first) * Width + col] = x[(Masks - 1) * Tile + col];
        #endif
        __syncthreads();
    }
}
