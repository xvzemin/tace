"""Convert EquFlashV2 FullConv interactions."""

from collections import OrderedDict

import torch
from e3nn import o3

from eqx.models.convolution import (
    Convolution,
    RadialFeatures,
    convert_modules,
    default_dtype,
)


class FullConv(torch.nn.Module):
    """Preserve EquFlashV2's convolution interface and output channel order."""

    def __init__(self, module, implementation, backend, normalization, normalize):
        super().__init__()
        import cuequivariance as cue

        if module.use_lammps_mliap:
            raise NotImplementedError(
                "Distributed ghost-atom exchange is not supported."
            )
        if len({mul for mul, _ in module.irreps_in}) != 1 or any(
            mul != 1 for mul, _ in module.irreps_filter
        ):
            raise NotImplementedError(
                "FullConv requires uniform input channels and unweighted harmonics."
            )
        poly = cue.descriptors.channelwise_tensor_product(
            cue.Irreps("O3", module.irreps_in),
            cue.Irreps("O3", module.irreps_filter),
            cue.Irreps("O3", module.irreps_out),
        )
        irreps_out = o3.Irreps(str(poly.outputs[0].irreps))
        if irreps_out.simplify() != module.irreps_out:
            raise ValueError(
                "FullConv output representations do not match the descriptor."
            )
        _, descriptor = poly.polynomial.operations[0]
        instructions = []
        for path in sorted(descriptor.paths, key=lambda p: p.indices[0]):
            _, i1, i2, iout = path.indices
            cg = o3.wigner_3j(
                module.irreps_in[i1].ir.l,
                module.irreps_filter[i2].ir.l,
                irreps_out[iout].ir.l,
                dtype=torch.float64,
            )
            coefficients = torch.tensor(path.coefficients, dtype=torch.float64)
            scale = float((cg * coefficients).sum() / cg.square().sum())
            if scale <= 0 or not torch.allclose(
                coefficients, scale * cg, atol=1e-12, rtol=1e-12
            ):
                raise NotImplementedError("The descriptor uses a different CG basis.")
            instructions.append((i1, i2, iout, "uvu", True, scale**2))
        tp = o3.TensorProduct(
            module.irreps_in,
            module.irreps_filter,
            irreps_out,
            instructions,
            irrep_normalization="none",
            path_normalization="none",
            internal_weights=False,
            shared_weights=False,
            compile_left_right=False,
        )
        if tp.weight_numel != module.weight_nn.hs[-1]:
            raise ValueError("FullConv radial weights do not match the descriptor.")
        self.convolution = Convolution(
            tp,
            module.weight_nn[-1],
            implementation=implementation,
            backend=backend,
            layout="ir_mul",
            normalization=normalization,
            normalize=normalize,
        )
        # Merge paths along channels after node aggregation, not on edges.
        indices = []
        for _, ir in irreps_out.simplify():
            for m in range(ir.dim):
                for (mul, other), section in zip(irreps_out, irreps_out.slices()):
                    if other == ir:
                        indices.extend(
                            range(
                                section.start + m * mul, section.start + (m + 1) * mul
                            )
                        )
        self.convolution.output_index = torch.tensor(indices, dtype=torch.long)
        self.weight_nn = RadialFeatures(
            OrderedDict(list(module.weight_nn.named_children())[:-1]),
            hs=module.weight_nn.hs[:-1],
        ).train(module.weight_nn.training)
        self.denominator = module.denominator
        self.irreps_in = module.irreps_in
        self.irreps_filter = module.irreps_filter
        self.irreps_out = module.irreps_out
        self.use_lammps_mliap = False

    def forward(self, x, edge_index, edge_embedding, edge_attr, lmp_data=None):
        if lmp_data is not None:
            raise NotImplementedError(
                "Distributed ghost-atom exchange is not supported."
            )
        return (
            self.convolution(
                x,
                edge_attr,
                self.weight_nn(edge_embedding),
                edge_index.flip(0).to(dtype=torch.long),
            )
            / self.denominator
        )


def convert_equflashv2_to_eqx(
    model,
    *,
    implementation="o3",
    inplace=False,
    backend="cuda",
):
    """Replace EquFlashV2 FullConv interactions with EQX convolutions.

    Parameters
    ----------
    model : torch.nn.Module
        Uncompiled model with uniform-channel FullConv interactions.
    implementation : {"o3", "o2"}, optional
        Direct or aligned tensor product. Defaults to "o3".
    inplace : bool, optional
        Modify the supplied model instead of returning a copy.
    backend : {"cuda", "torch"}, optional
        Convolution backend. Defaults to CUDA, with PyTorch on CPU.

    Returns
    -------
    torch.nn.Module
        Model with the original forward interface and differentiable parameters.
        Load original weights before conversion and construct the optimizer after it.
    """
    if implementation not in ("o3", "o2") or backend not in ("cuda", "torch"):
        raise ValueError(
            "Expected implementation 'o3'/'o2' and backend 'cuda'/'torch'."
        )
    options = {
        (m.normalization, m.normalize)
        for m in model.modules()
        if isinstance(m, o3.SphericalHarmonics)
    }
    if len(options) > 1:
        raise NotImplementedError(
            "Multiple harmonic conventions require separate conversion."
        )
    normalization, normalize = next(iter(options), ("component", True))
    count = 0

    def factory(module):
        nonlocal count
        if isinstance(module, FullConv):
            count += 1
            if (module.convolution.implementation, module.convolution.backend) != (
                implementation,
                backend,
            ):
                raise ValueError(
                    "Convert the original model to select another implementation."
                )
            return module
        if not type(module).__module__.startswith("GGNN.model.EquFlashV2."):
            return None
        if type(module).__name__ == "EfficientConv":
            raise NotImplementedError("Only EquFlashV2 FullConv is supported.")
        if type(module).__name__ != "FullConv":
            return None
        count += 1
        parameter = module.weight_nn[-1].weight
        with default_dtype(parameter.dtype):
            return (
                FullConv(
                    module,
                    implementation,
                    backend,
                    normalization,
                    normalize,
                )
                .to(parameter.device)
                .train(module.training)
            )

    converted = convert_modules(model, factory, inplace=inplace)
    if not count:
        raise ValueError("No EquFlashV2 FullConv interactions were found.")
    return converted
