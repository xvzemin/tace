"""Convert EquFlash FullConv interactions."""

from collections import OrderedDict
from itertools import groupby

import torch
from e3nn import o3

from ...utils.module import default_dtype
from ..convolution import (
    Convolution,
    RadialFeatures,
    convert_interactions,
    harmonic_convention,
    spatial_backend,
)


class FullConv(torch.nn.Module):
    """Preserve EquFlash's convolution interface and output channel order."""

    def __init__(
        self, module, implementation, backend, normalization, normalize, stream_radial
    ):
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
            radial_mode="full" if stream_radial else "last",
        )
        # Merge paths along channels after node aggregation, not on edges.
        indices = []
        sections = (
            (mul, ir, s) for (mul, ir), s in zip(irreps_out, irreps_out.slices()) if mul
        )
        for ir, group in groupby(sections, key=lambda entry: entry[1]):
            parts = list(group)
            for m in range(ir.dim):
                for mul, _, section in parts:
                    indices.extend(
                        range(section.start + m * mul, section.start + (m + 1) * mul)
                    )
        self.convolution.set_permutation(
            "output_index", torch.tensor(indices, dtype=torch.long)
        )
        self.weight_nn = RadialFeatures(
            OrderedDict(list(module.weight_nn.named_children())[:-1]),
            hs=module.weight_nn.hs[:-1],
            stream_radial=stream_radial,
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
        target, source = edge_index.to(dtype=torch.long)
        return (
            self.convolution(
                x,
                edge_attr,
                self.weight_nn(edge_embedding),
                (source, target),
            )
            / self.denominator
        )


def convert_equflash_to_eqx(
    model,
    *,
    implementation="o3",
    inplace=False,
    backend=None,
    stream_radial=False,
):
    """Replace EquFlash FullConv interactions with EQX convolutions.

    Parameters
    ----------
    model : torch.nn.Module
        Uncompiled EquFlashV2 model with uniform-channel FullConv interactions.
        EfficientConv and distributed ghost exchange are not supported.
    implementation : {"o3", "o2"}, optional
        Direct or aligned tensor product. Defaults to "o3".
    inplace : bool, optional
        Modify the supplied model instead of returning a copy. Use True with UCalculator.
    backend : {None, "cuda", "torch"}, optional
        Execution backend. Defaults to CUDA when available, otherwise PyTorch.
    stream_radial : bool, optional
        Fuse the complete radial MLP if True, or its final projection if False.

    Returns
    -------
    torch.nn.Module
        Converted model with the original forward interface.

    Examples
    --------
    .. code-block:: python

        from equivariantx.models.equflash import convert_equflash_to_eqx

        model = convert_equflash_to_eqx(model, implementation="o3")
    """
    backend = spatial_backend(implementation, backend)
    normalization, normalize = (
        harmonic_convention(model) if implementation == "o2" else ("component", True)
    )

    def factory(module):
        if isinstance(module, FullConv):
            module.convolution.check_options(
                module.weight_nn, implementation, backend, stream_radial
            )
            return module
        if not type(module).__module__.startswith("GGNN.model.EquFlashV2."):
            return None
        if type(module).__name__ == "EfficientConv":
            raise NotImplementedError("Only EquFlash FullConv is supported.")
        if type(module).__name__ != "FullConv":
            return None

        parameter = module.weight_nn[-1].weight
        with default_dtype(parameter.dtype):
            return (
                FullConv(
                    module,
                    implementation,
                    backend,
                    normalization,
                    normalize,
                    stream_radial,
                )
                .to(parameter.device)
                .train(module.training)
            )

    return convert_interactions(
        model, factory, inplace=inplace, name="EquFlash FullConv interactions"
    )
