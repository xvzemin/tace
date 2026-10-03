"""Tensor-product adapters shared by atomistic models."""

import math

import torch

from eqx.conv import O2O3TensorProductConv, O3TensorProductConv
from eqx.kernels import wigner_D
from eqx.o2 import O3TensorProduct, WignerD


class RadialFeatures(torch.nn.Sequential):
    """Evaluate radial layers preceding an affine projection."""

    def __init__(self, layers, bias=False, hs=()):
        super().__init__(layers)
        self.bias = bias
        self.hs = list(hs)

    def forward(self, inputs):
        features = super().forward(inputs)
        if self.bias:
            features = torch.cat((features, torch.ones_like(features[:, :1])), dim=-1)
        return features


class Convolution(torch.nn.Module):
    """Adapt feature layouts and radial projections to EQX convolutions.

    Parameters
    ----------
    tensor_product : e3nn.o3.TensorProduct
        Tensor product with external ``uvu`` weights.
    projection : torch.nn.Module or None
        Final affine radial layer, or None for precomputed path weights.
    implementation : {"o3", "o2"}
        Direct or aligned tensor product.
    backend : {"cuda", "torch"}
        Kernel backend. CPU tensors use PyTorch.
    layout : {"mul_ir", "ir_mul"}
        Input and output feature storage.
    normalization : {"component", "integral", "norm"}
        Edge spherical-harmonic normalization for the aligned form.
    normalize : bool
        Whether edge harmonics are evaluated on unit vectors.
    packed_dim : int
        Packed Wigner entries appended after edge harmonics, if supplied.
    """

    def __init__(
        self,
        tensor_product,
        projection=None,
        *,
        implementation="o3",
        backend="cuda",
        layout="mul_ir",
        normalization="component",
        normalize=True,
        packed_dim=0,
    ):
        super().__init__()
        if implementation not in ("o3", "o2"):
            raise ValueError("implementation must be 'o3' or 'o2'.")
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError("layout must be 'mul_ir' or 'ir_mul'.")
        self.implementation = implementation
        self.backend = backend
        self.projection = projection
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.weight_numel = tensor_product.weight_numel
        self.harmonic_dim = self.irreps_in2.dim
        self.packed_dim = packed_dim
        self.normalize = normalize
        if implementation == "o2":
            # path_weight is already normalized in the source instructions.
            instructions = [
                (
                    ins.i_in1,
                    ins.i_in2,
                    ins.i_out,
                    ins.connection_mode,
                    ins.has_weight,
                    ins.path_weight**2,
                )
                for ins in tensor_product.instructions
            ]
            tensor_product = O3TensorProduct(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                instructions,
                irrep_normalization="none",
                path_normalization="none",
                internal_weights=False,
                shared_weights=False,
                normalization=normalization,
            )
            self.eqx_tp = O2O3TensorProductConv(tensor_product, backend=backend)
            if not packed_dim:
                self.wigner = WignerD(tensor_product.lmax, tensor_product.lmax)
            self.direction_start = next(
                (
                    section.start
                    for (_, ir), section in zip(
                        self.irreps_in2, self.irreps_in2.slices()
                    )
                    if ir.l == 1
                ),
                -1,
            )
            if not packed_dim and self.direction_start < 0 and self.irreps_in2.lmax:
                raise NotImplementedError(
                    "Aligned conversion requires degree-one harmonics."
                )
            self.direction_scale = {
                "component": math.sqrt(3),
                "integral": math.sqrt(3 / (4 * math.pi)),
                "norm": 1.0,
            }[normalization]
        else:
            self.eqx_tp = O3TensorProductConv(tensor_product, backend=backend)

        for name, irreps, inverse in (
            ("input_index", self.irreps_in1, False),
            ("attribute_index", self.irreps_in2, False),
            ("output_index", self.irreps_out, True),
        ):
            index = []
            for (mul, ir), section in zip(irreps, irreps.slices()):
                index.extend(
                    section.start + channel * ir.dim + m
                    for m in range(ir.dim)
                    for channel in range(mul)
                )
            index = torch.tensor(index, dtype=torch.long)
            if inverse:
                index = index.argsort()
            if layout == "ir_mul" or torch.equal(index, torch.arange(irreps.dim)):
                index = torch.empty(0, dtype=torch.long)
            self.register_buffer(name, index, persistent=False)
        self.register_buffer(
            "amplitudes", torch.ones(1, len(self.irreps_in2)), persistent=False
        )
        self.register_buffer(
            "degrees",
            torch.tensor([ir.l for _, ir in self.irreps_in2]),
            persistent=False,
        )
        self.register_buffer(
            "empty_projection", torch.empty(0, self.weight_numel), persistent=False
        )
        self.affine = isinstance(projection, torch.nn.Linear)
        self.projection_scale = 1.0
        if projection is not None and hasattr(projection, "h_in"):
            if projection.act is not None:
                raise NotImplementedError("The final radial projection must be linear.")
            self.projection_scale = (
                projection.h_in * projection.var_in / projection.var_out
            ) ** -0.5

    def forward(self, node_feats, edge_attrs, radial, edge_index):
        if self.input_index.numel():
            node_feats = node_feats.index_select(-1, self.input_index)
        projection = self.empty_projection
        if self.projection is not None:
            if self.affine:
                projection = self.projection.weight.T
            elif hasattr(self.projection, "weights"):
                projection = self.projection.weights
            else:
                projection = self.projection.weight * self.projection_scale
                if hasattr(self.projection, "alpha"):
                    projection = projection * self.projection.alpha
            bias = getattr(self.projection, "bias", None)
            if bias is not None:
                projection = torch.cat((projection, bias[None]), dim=0)
        if self.implementation == "o3":
            attributes = edge_attrs[:, : self.harmonic_dim]
            if self.attribute_index.numel():
                attributes = attributes.index_select(-1, self.attribute_index)
            message = self.eqx_tp(
                node_feats.contiguous(),
                attributes.contiguous(),
                radial.contiguous(),
                projection,
                edge_index,
            )
        else:
            vectors = None
            if self.packed_dim:
                end = self.harmonic_dim + self.packed_dim
                packed = edge_attrs[:, self.harmonic_dim : end].contiguous()
                amplitudes = (
                    self.amplitudes
                    if self.normalize
                    else edge_attrs[:, end:].contiguous()
                )
            else:
                if self.direction_start >= 0:
                    vectors = (
                        edge_attrs[:, self.direction_start : self.direction_start + 3]
                        / self.direction_scale
                    )
                else:
                    vectors = edge_attrs.new_tensor([0.0, 1.0, 0.0]).expand(
                        edge_attrs.size(0), 3
                    )
                packed = wigner_D(self.wigner, vectors, backend=self.backend)
                amplitudes = (
                    self.amplitudes
                    if self.normalize
                    else vectors.norm(dim=-1, keepdim=True).pow(self.degrees)
                )
            message = self.eqx_tp(
                node_feats.contiguous(),
                radial.contiguous(),
                projection,
                packed,
                amplitudes,
                edge_index,
                node_feats.size(0),
                vectors=vectors,
            )
        if self.output_index.numel():
            message = message.index_select(-1, self.output_index)
        return message
