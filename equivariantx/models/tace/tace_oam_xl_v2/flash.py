"""Optional FlashTP bridge for an unchanged OAM convolution contract.

FlashTP is an external MIT-licensed library (SNU-ARC/flashTP, ICML 2025).
This adapter preserves TACE's native mul_ir layout and keeps static metadata as nonpersistent
buffers. It introduces no trainable parameters or checkpoint keys.
"""

import torch

from equivariantx.utils.metadata import parse_metadata


class FlashConvolution(torch.nn.Module):
    """Run the exact retained UVU paths with FlashTP's mixed derivatives."""

    def __init__(self, metadata, device, dtype=torch.float32):
        super().__init__()
        import flashTP_e3nn
        from e3nn import o3

        if dtype != torch.float32:
            raise ValueError("The FlashTP bridge currently validates FP32 execution")
        paths, width, shared = parse_metadata(metadata)
        if shared or any(p[4] != 1 for p in paths):
            raise ValueError(
                "FlashTP requires unshared radial weights and scalar harmonics"
            )
        self.input_blocks = tuple(sorted(set((p[0], p[3], p[5]) for p in paths)))
        angles = sorted(set((p[1], p[4], p[6]) for p in paths))
        self.output_blocks = tuple(sorted(set((p[2], p[3], p[7]) for p in paths)))

        def irreps(blocks):
            return o3.Irreps(
                [(mul, ((d - 1) // 2, (-1) ** ((d - 1) // 2))) for _, mul, d in blocks]
            )

        ix, iy, iz = (
            irreps(self.input_blocks),
            irreps(angles),
            irreps(self.output_blocks),
        )
        instructions = [
            (
                self.input_blocks.index((p[0], p[3], p[5])),
                angles.index((p[1], p[4], p[6])),
                self.output_blocks.index((p[2], p[3], p[7])),
                "uvu",
                True,
            )
            for p in paths
        ]
        standard = o3.TensorProduct(
            ix, iy, iz, instructions, shared_weights=False, internal_weights=False
        )
        scale = torch.empty(width, device=device, dtype=dtype)
        offset = 0
        for path, inst in zip(paths, standard.instructions):
            if offset != path[8]:
                raise ValueError("Unexpected radial weight path order")
            scale[offset : offset + path[3]] = path[9] / inst.path_weight
            offset += path[3]
        self.register_buffer("scale", scale, persistent=False)
        self.identity_scale = bool(torch.all(scale == 1))
        native = flashTP_e3nn.uvu_TP(
            ix,
            iy,
            iz,
            instructions,
            block_batch_cnt=[4, 2, 2],
            smem_size=227,
            device=device,
            dtype=dtype,
        )
        self.metadata_count = len(native.metadata_list)
        for index, value in enumerate(native.metadata_list):
            self.register_buffer(f"metadata_{index}", value, persistent=False)
        self.paths = native.uppath_cnt
        self.blocks = native.per_block_batch
        self.lmax = native.l_max
        self.output_dim = native.out_dim

    def forward(self, nodes, attrs, weights, source, target):
        radial = weights if self.identity_scale else weights * self.scale
        return torch.ops.flashtp_large_kernel.sptp_linear_fwd_v2_shared_exp(
            nodes,
            attrs,
            radial,
            source,
            target,
            *(getattr(self, f"metadata_{i}") for i in range(self.metadata_count)),
            self.paths,
            self.blocks,
            self.lmax,
            self.output_dim,
        )
