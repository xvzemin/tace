"""Order-wise channel contractions for large packed local O(2) convolutions."""

from functools import lru_cache

import torch

from ..codegen import HEADER
from ..graph import capture_constant
from ..runtime import kernels, runtime


@lru_cache(maxsize=64)
def supports(plan):
    """Return whether the convolution has uniform, independently weighted paths."""
    if not plan.path_data or plan.has_unweighted or plan.scalar_paths:
        return False
    width = plan.path_data[0][1][2]
    return width > 0 and all(
        mode == "uvu"
        and p[2] == p[3] == width
        and min(p[6:9]) >= 0
        and p[9] == 0
        and not (p[0] % width or p[1] % width or p[8] % width)
        for mode, p in plan.path_data
    )


def frame_table(blocks, width):
    """Index global blocks, local rows and entries of the supplied Wigner matrices."""
    positions, angular, matrix = {}, [], []
    for start, dim, offset in sorted(set(blocks)):
        base = len(angular)
        positions[start, dim, offset] = base
        angular.extend([(start, offset, dim, base)] * dim)
        matrix.extend(
            (start + b * width, base + a, offset + a * dim + b, 0)
            for a in range(dim)
            for b in range(dim)
        )
    return positions, (angular, matrix)


@lru_cache(maxsize=32)
def tables(plan, dtype, device):
    """Build three sparse local transposes and two indexed rotation layouts."""
    import numpy as np

    width = plan.path_data[0][1][2]
    paths = [p for _, p in plan.path_data]
    xi, xframe = frame_table(((p[0], p[4], p[6]) for p in paths), width)
    yi, yframe = frame_table(((p[1], p[5], p[7]) for p in paths), width)
    dims = (len(xframe[0]), plan.weight_numel // width, len(yframe[0]))
    rows, weights = [], []
    for p, coeffs in zip(paths, plan.sparse_paths):
        x, y = xi[p[0], p[4], p[6]], yi[p[1], p[5], p[7]]
        for a, b, c in coeffs:
            rows.append((x + a, p[8] // width, y + b))
            weights.append(c)
    rows = np.asarray(rows, dtype=np.int32).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64)

    def integers(x):
        return torch.as_tensor(np.ascontiguousarray(x, dtype=np.int32), device=device)

    csr = []
    for role, dim in enumerate(dims):
        order = np.argsort(rows[:, role], kind="stable")
        offsets = np.empty(dim + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(np.bincount(rows[:, role], minlength=dim), out=offsets[1:])
        others = [j for j in range(3) if j != role]
        csr.append(
            (
                integers(offsets),
                integers(rows[order][:, others]),
                torch.as_tensor(weights[order], device=device, dtype=dtype),
            )
        )
    frames = tuple(
        tuple(integers(part) for part in frame) for frame in (xframe, yframe)
    )
    return width, dims, frames, tuple(csr)


@lru_cache(maxsize=2)
def sources(dtype):
    """Generate degree-independent CUDA programs for the trilinear pairing."""
    header = HEADER.replace("SCALAR", "double" if dtype == torch.float64 else "float")
    header += "using I=long long; struct B {int x,y,z,w;}; struct P {int x,y;};\n"
    result = {}
    for name in ("rotate", "scatter"):
        left, right = (
            ("node[n*ns+v.x+j*C+c]", "matrix[e*ms+v.y+r*v.z+j]")
            if name == "rotate"
            else ("local[(e*dim+v.w+j)*C+c]", "matrix[e*ms+v.y+j*v.z+r]")
        )
        store = (
            "out[e*dim*C+i]=sum;"
            if name == "rotate"
            else "atomicAdd(out+n*ns+v.x+r*C+c,sum*scale[e*ss]);"
        )
        result[name] = (
            header
            + f"""
        extern "C" __global__ void run(const B* blocks,const I* index,const T* node,
            const T* matrix,const T* local,const T* scale,T* out,I ns,I ms,I ss,I dim,I C){{
          I e=blockIdx.x,n=index[e];
          for(I i=(I)blockIdx.y*blockDim.x+threadIdx.x;i<dim*C;i+=(I)gridDim.y*blockDim.x){{
            I a=i/C,c=i%C; B v=blocks[a]; I r=a-v.w; T sum=0;
            for(int j=0;j<v.z;++j) sum=fma({left},{right},sum);
            {store}
          }}
        }}"""
        )
    result["matrix"] = (
        header
        + """
        extern "C" __global__ void run(const B* entries,const I* index,const T* node,
            const T* local,const T* scale,T* out,I ns,I ls,I ms,I ss,I count,I C){
          I e=blockIdx.x,n=index[e]; int lane=threadIdx.x%32;
          for(I i=(I)blockIdx.y*(blockDim.x/32)+threadIdx.x/32;i<count;i+=(I)gridDim.y*(blockDim.x/32)){
            B v=entries[i]; T sum=0;
            for(int c=lane;c<C;c+=32) sum=fma(node[n*ns+v.x+c],local[e*ls+v.y*C+c],sum);
            sum=warp_sum(sum);
            if(!lane) atomicAdd(out+e*ms+v.z,sum*scale[e*ss]);
          }
        }"""
    )
    result["local"] = (
        header
        + """
        extern "C" __global__ void run(const int* ptr,const P* paths,const T* coeff,
            const T* x,const T* y,const T* scale,T* out,I xs,I ys,I ss,I os,
            I dim,I C,I scaled,I accumulate){
          I e=blockIdx.x;
          for(I i=(I)blockIdx.y*blockDim.x+threadIdx.x;i<dim*C;i+=(I)gridDim.y*blockDim.x){
            I a=i/C,c=i%C; T sum=0;
            for(int j=ptr[a];j<ptr[a+1];++j){
              P p=paths[j]; sum=fma(coeff[j]*x[e*xs+p.x*C+c],y[e*ys+p.y*C+c],sum);
            }
            if(scaled) sum*=scale[e*ss];
            if(!os) atomicAdd(out+i,sum);
            else if(accumulate) out[e*os+i]+=sum;
            else out[e*os+i]=sum;
          }
        }"""
    )
    result["amplitude"] = (
        header
        + """
        extern "C" __global__ void run(const T* x,const T* y,T* out,I dim,I os){
          __shared__ T sums[4]; I e=blockIdx.x; T sum=0;
          for(I i=threadIdx.x;i<dim;i+=128) sum=fma(x[e*dim+i],y[e*dim+i],sum);
          sum=warp_sum(sum);
          if(!(threadIdx.x%32)) sums[threadIdx.x/32]=sum;
          __syncthreads();
          if(!threadIdx.x) atomicAdd(out+e*os,(sums[0]+sums[1])+(sums[2]+sums[3]));
        }"""
    )
    return result


def contract(plan, source, target, calls, shared=None, initialize=()):
    """Rotate once per operand and edge tile, retaining every independent path."""
    sample = calls[0][1][0]
    width, dims, frames, csr = capture_constant(
        tables, plan, sample.dtype, sample.device
    )
    spec = sources(sample.dtype)
    binaries = kernels(spec.values(), sample.device)
    compiled = {name: binaries[code] for name, code in spec.items()}
    stream = torch.cuda.current_stream(sample.device).cuda_stream
    shared = shared or tuple(calls[0][1][i].size(0) == 1 for i in (1, 3, 4, 5))
    initial = {id(value) for value in initialize}
    # Only rotations and local results are temporary. The wide radial adjoint
    # is written directly to its final buffer, including projected-weight tiles.
    live = 2 * (dims[0] + dims[2]) * width * max(1, len(calls))
    tile = max(1, (192 << 20) // (max(live, 1) * sample.element_size()))
    tile = max(1, tile // 32 * 32) if tile >= 32 else tile
    for begin in range(0, source.numel(), tile):
        end = min(source.numel(), begin + tile)
        size = end - begin
        jobs, cache, written = [], {}, set()
        indices = (source[begin:end], target[begin:end])

        def launch(name, tensors, args, tasks, lanes=128):
            jobs.append(
                (
                    compiled[name],
                    [t.data_ptr() for t in tensors] + list(args),
                    size,
                    min(65535, (tasks + lanes - 1) // lanes),
                    128,
                    0,
                )
            )

        def view(t, common):
            return t if common else t[begin:end]

        def stride(t, common):
            return 0 if common else t.size(1)

        for roles, values, results, _ in calls:
            x, w, _, di, do, s, y = values
            weight, scale = view(w, shared[0]), view(s, shared[3])
            nodes, matrices = (x, y), (di, do)
            ss = stride(s, shared[3])

            def rotation(side):
                n, d = nodes[side], matrices[side]
                key = "r", side, id(n), id(d)
                if key not in cache:
                    dim = dims[2 * side]
                    out = sample.new_empty((size, dim * width))
                    launch(
                        "rotate",
                        (
                            frames[side][0],
                            indices[side],
                            n,
                            view(d, shared[side + 1]),
                            out,
                            scale,
                            out,
                        ),
                        (n.size(1), stride(d, shared[side + 1]), ss, dim, width),
                        dim * width,
                    )
                    cache[key] = out
                return cache[key]

            def product(side):
                other = 1 - side
                key = "p", side, id(w), id(nodes[other]), id(matrices[other])
                if key not in cache:
                    local = rotation(other)
                    lhs, rhs = (weight, local) if side == 0 else (local, weight)
                    xs, ys = (
                        (stride(w, shared[0]), local.size(1))
                        if side == 0
                        else (local.size(1), stride(w, shared[0]))
                    )
                    dim = dims[2 * side]
                    out = sample.new_empty((size, dim * width))
                    launch(
                        "local",
                        (*csr[2 * side], lhs, rhs, scale, out),
                        (xs, ys, ss, out.size(1), dim, width, 0, 0),
                        dim * width,
                    )
                    cache[key] = out
                return cache[key]

            for role, out in zip(roles, results):
                if role in (0, 6):
                    side = int(role == 6)
                    d = matrices[side]
                    local = product(side)
                    dim = dims[2 * side]
                    launch(
                        "scatter",
                        (
                            frames[side][0],
                            indices[side],
                            out,
                            view(d, shared[side + 1]),
                            local,
                            scale,
                            out,
                        ),
                        (out.size(1), stride(d, shared[side + 1]), ss, dim, width),
                        dim * width,
                    )
                elif role in (3, 4):
                    side = role - 3
                    n, local, table = nodes[side], product(side), frames[side][1]
                    launch(
                        "matrix",
                        (
                            table,
                            indices[side],
                            n,
                            local,
                            scale,
                            view(out, shared[side + 1]),
                        ),
                        (
                            n.size(1),
                            local.size(1),
                            stride(out, shared[side + 1]),
                            ss,
                            table.size(0),
                            width,
                        ),
                        table.size(0),
                        4,
                    )
                elif role == 1:
                    lx, ly = rotation(0), rotation(1)
                    dim = dims[1]
                    launch(
                        "local",
                        (*csr[1], lx, ly, scale, view(out, shared[0])),
                        (
                            lx.size(1),
                            ly.size(1),
                            ss,
                            stride(out, shared[0]),
                            dim,
                            width,
                            1,
                            int(id(out) not in initial or id(out) in written),
                        ),
                        dim * width,
                    )
                    written.add(id(out))
                elif role == 5:
                    local, ly = product(1), rotation(1)
                    launch(
                        "amplitude",
                        (local, ly, view(out, shared[3])),
                        (local.size(1), ss),
                        1,
                    )
                else:
                    raise ValueError(
                        "Project radial weights before the order contraction"
                    )
        runtime().launch(jobs, stream)
