"""Execution choices for TACE property heads."""

import math
import sys

import torch
from e3nn import o3

from equivariantx import co3
from equivariantx.utils.autograd import forward_ad_active
from tace.models.spherical import edge_inter
from tace.models.spherical.fused import UvuTensorProduct
from tace.models.spherical.paths import generate_paths
from tace.models.spherical.representation import PairRepresentation
from tace.models.spherical.tace import SphericalTACE

from .matching import custom_forward, tracing_rows
from .planner import Region


def _host(path, reason, provider=None):
    return Region(
        path,
        "native",
        reason,
        provenance="host_selected" if provider else "host_fallback",
        provider=provider,
        scope="readout_fn",
    )


def _standard_readout(host):
    """The host still calls the configured heads through the standard TACE methods."""
    return (
        isinstance(host, SphericalTACE)
        and type(host).forward is SphericalTACE.forward
        and type(host).readout_fn is SphericalTACE.readout_fn
        and not {"forward", "readout_fn"}.intersection(vars(host))
    )


def _compact_periodic_grid(ewald):
    start = ewald.hemisphere_mask.numel() // 2
    for name in ("nvec_all", "hemisphere_mask", "factors"):
        setattr(ewald, name, getattr(ewald, name)[start:].clone())


def _canonical_periodic_grid(ewald):
    """Prove the three live, nonpersistent LES 0.2.0 tables before slicing."""
    if type(ewald.N_max) is not int or ewald.N_max < 1:
        return False
    side = 2 * ewald.N_max + 1
    count = side**3
    middle = count // 2
    names = ("nvec_all", "hemisphere_mask", "factors")
    grid, mask, factors = (ewald._buffers.get(n) for n in names)
    if not (
        all(
            type(t) is torch.Tensor
            and not t.is_meta
            and not t.requires_grad
            and t.is_contiguous()
            for t in (grid, mask, factors)
        )
        and set(names) <= ewald._non_persistent_buffers_set
        and grid.device == mask.device == factors.device
        and grid.dtype == torch.int64
        and mask.dtype == torch.bool
        and factors.is_floating_point()
        and grid.shape == (count, 3)
        and mask.shape == factors.shape == (count,)
    ):
        return False
    # Compare against broadcast views, not a second O(N_max**3) vector grid.
    # Check the actual values: buffers may be loaded or edited in place.
    axis = torch.arange(-ewald.N_max, ewald.N_max + 1, device=grid.device)
    vectors = grid.view(side, side, side, 3).unbind(-1)
    return (
        all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                vectors, torch.meshgrid(axis, axis, axis, indexing="ij")
            )
        )
        and torch.equal(mask[:middle], mask.new_tensor(False).expand(middle))
        and torch.equal(mask[middle:], mask.new_tensor(True).expand(middle + 1))
        and torch.equal(factors[middle], factors.new_tensor(1))
        and all(
            torch.equal(part, factors.new_tensor(2).expand(middle))
            for part in (factors[:middle], factors[middle + 1 :])
        )
    )


def _automatic_periodic_grid(model, ewald):
    """The measured, single-device E/F/S charge solver; checked only at planning."""
    from tace.models.adapter import TensorModel
    from tace.utils.env import memory_first

    if (
        type(model) is not TensorModel
        or not _standard_readout(model.readout_fn)
        or model.lmp
        or not memory_first()
        or custom_forward(model)
        or model.get_model_dtype() != torch.float32
        or type(ewald._buffers.get("nvec_all")) is not torch.Tensor
        or not ewald.nvec_all.is_cuda
        or type(ewald._buffers.get("factors")) is not torch.Tensor
        or torch.is_autocast_enabled("cuda")
        or torch.get_float32_matmul_precision() not in ("high", "highest")
    ):
        return False
    host = model.readout_fn
    cfg = host.model_config
    return (
        not custom_forward(host)
        and set(host.target_property) == {"energy", "forces", "stress"}
        and not model.get_embedding_property()
        and not cfg.get("embedding_property")
        and not any(
            v.get("enable") for v in cfg.get("universal_embedding", {}).values()
        )
        and not getattr(host.representation, "use_magnetic_interaction", True)
        and type(ewald.N_max) is int
        and ewald.N_max == 10
        and all(
            type(actual) in (int, float) and actual == expected
            for actual, expected in (
                (ewald.dl, 2.0),
                (ewald.sigma, 1.0),
                (ewald.k_sq_max, math.pi**2),
                (ewald.sigma_sq_half, 0.5),
            )
        )
        and ewald.remove_self_interaction is True
        and ewald.factors.dtype == torch.float64
        and all(p.device == ewald.nvec_all.device for p in model.parameters())
        and all(p.dtype == torch.float32 for p in host.les.parameters())
    )


def periodic_les_region(head, *, automatic, cuda_model=None):
    """Select the reciprocal hemisphere (including its masked zero vector)."""
    from tace.models.spherical.les import is_periodic_charge_only_les

    if not is_periodic_charge_only_les(head) or custom_forward(head, recursive=True):
        return None
    ewald = head.les.ewald
    reason = "LES periodic charge sum on the retained reciprocal hemisphere"
    installed = getattr(ewald, "_foundation_regional_provider", None)
    if installed is None:
        if automatic and not _automatic_periodic_grid(cuda_model, ewald):
            return None
        if not _canonical_periodic_grid(ewald):
            return None
    return Region(
        "les.les.ewald",
        "torch",
        reason,
        lower=None if installed else _compact_periodic_grid,
        provenance="already_regional" if installed else "regional",
        provider="les_torch",
        scope="readout_fn",
    )


def _pair_interaction(layer):
    if custom_forward(layer, recursive=True):
        return "Preserve the customized host pair interaction.", None
    if type(layer) is edge_inter.O2EdgeInteraction:
        return "Host O(2) pair map; frame, linear and gate dispatch separately.", None
    if type(layer) is not edge_inter.O3CgtpEdgeInteraction:
        return "The host owns this pair interaction.", None
    product = layer.rejector
    rowwise = getattr(edge_inter, "_RowwisePairProduct", None)
    if type(product) not in (UvuTensorProduct, rowwise):
        return "Preserve the host pair tensor product.", None
    reason = "Host directed-edge CG product."
    param = next(layer.parameters(), None)
    cuda_float = (
        param is not None
        and param.is_cuda
        and param.dtype in (torch.float32, torch.float64)
    )
    if type(product) is rowwise:
        from equivariantx.o3 import O3TensorProductConv

        if type(product.tp) is not o3.TensorProduct:
            return (
                "Nonstandard host row-wise product; provider is not classified.",
                None,
            )
        reason = (
            "Host row-wise CG branch requires matching FP32/FP64 CUDA rows and "
            "autocast off; other inputs use the host tensor product."
        )
        canonical = getattr(product, "_canonical_cg", None)
        if cuda_float and canonical is not None and not canonical():
            return "Host e3nn reads live or noncanonical CG tables.", "e3nn"
        if cuda_float:
            op = getattr(product, "contraction", None)
            if type(op) is O3TensorProductConv and op.backend in ("torch", "cuda"):
                return reason, f"eqx_{op.backend}"
            return reason, None
    if hasattr(product, "fused_tp"):
        # Do not import optional GPU extensions just to generate a plan.
        wrapper = sys.modules.get("tace.models.backends.openequivariance")
        provider = (
            "oeq"
            if cuda_float
            and type(product.fused_tp) is getattr(wrapper, "OeqTensorProduct", None)
            else None
        )
        return reason, provider
    op = product.tp
    if (
        type(op) is co3.TensorProduct
        and param is not None
        and op.backend in ("torch", "cuda")
    ):
        backend = op.backend if param.is_cuda else "torch"
        return "Host Cartesian ICTD pair tensor product.", f"eqx_{backend}"
    return reason, "e3nn" if type(op) is o3.TensorProduct else None


class _InferencePairProduct(
    getattr(edge_inter, "_RowwisePairProduct", UvuTensorProduct)
):
    """Reuse the host row-wise kernel only for the verified eager FP32 signature."""

    def _canonical_cg(self):
        # Our forward already certified the same live tables and e3nn hooks.
        return True

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Never retain a tensor on its old device, or assume replacement tables
        # still match the constants used to construct the CUDA program.
        if not self._unchanged_tables():
            self._pair_tables = ()
        return self

    def _unchanged_tables(self):
        if not self._pair_tables or type(self.tp) is not o3.TensorProduct:
            return False
        live = dict(self.tp.named_buffers())
        return all(
            live.get(n) is value and not value.requires_grad and value._version == v
            for n, value, v in self._pair_tables
        )

    def forward(self, x, y, weights):
        if (
            not tracing_rows(x)
            and not torch.jit.is_tracing()
            and not forward_ad_active()
            and not self.training
            and x.is_cuda
            and x.dtype == torch.float32
            and x.ndim == 2
            and x.shape[0] > 0
            and not hasattr(self, "fused_tp")
            and not custom_forward(self.tp, recursive=True)
            and self._unchanged_tables()
        ):
            return super().forward(x, y, weights)
        return UvuTensorProduct.forward(self, x, y, weights)


def _ordinary_pair_paths(p):
    """Match the measured full-order layout and the native CG constants it uses."""
    tp = p.tp
    if (
        type(tp) is not o3.TensorProduct
        or tp.internal_weights
        or tp.shared_weights
        or hasattr(p, "fused_tp")
        or any(True for _ in tp.parameters())
        or tp.irreps_in1 != o3.Irreps("64x0e+64x1o+64x1e+64x2e")
        or tp.irreps_in2 != o3.Irreps.spherical_harmonics(3)
        or tp.irreps_out.simplify() != o3.Irreps("192x0e+384x1o+256x1e+384x2e")
        or (p.irreps_in1, p.irreps_in2, p.irreps_out, p.weight_numel)
        != (tp.irreps_in1, tp.irreps_in2, tp.irreps_out, tp.weight_numel)
    ):
        return False
    instructions, out = generate_paths(
        tp.irreps_out.simplify(), tp.irreps_in1, tp.irreps_in2
    )
    if (
        out != tp.irreps_out
        or len(instructions) != len(tp.instructions)
        or any(
            tuple(i[:5]) != expected or i.path_weight != math.sqrt(out[i.i_out].ir.dim)
            for i, expected in zip(tp.instructions, instructions, strict=True)
        )
    ):
        return False
    for name, value in tp.named_buffers():
        if value.requires_grad or value.is_inference():
            return False
        if name in ("weight", "output_mask"):
            continue
        prefix = "_compiled_main_left_right._w3j_"
        if not name.startswith(prefix):
            return False
        degrees = name[len(prefix) :].split("_")
        if len(degrees) != 3 or not all(n.isdecimal() for n in degrees):
            return False
        canonical = o3.wigner_3j(*map(int, degrees), dtype=value.dtype, device="cpu")
        if not torch.equal(value.detach().cpu(), canonical):
            return False
    return True


def _install_pair(layer):
    p = layer.rejector
    p.__class__ = _InferencePairProduct
    p._setup_rowwise()
    like = next(layer.parameters())
    for part in (p.contraction, p.to_channels_last, p.to_mul_ir):
        part.to(like)
    p._pair_tables = tuple((n, b, b._version) for n, b in p.tp.named_buffers())
    return layer


def _restore_pair(layer):
    p = layer.rejector
    for name in ("contraction", "to_channels_last", "to_mul_ir", "_pair_tables"):
        delattr(p, name)
    p.__class__ = UvuTensorProduct
    return layer


def pair_interaction_region(host, *, backend, task, automatic):
    """Only explicit CUDA inference opts into the measured spherical HIP head."""
    pair = getattr(host, "pair_representation", None)
    if (
        type(host) is not SphericalTACE
        or type(pair) is not PairRepresentation
        or len(pair.interactions) != 1
    ):
        return None
    layer = pair.interactions[0]
    if type(layer) is not edge_inter.O3CgtpEdgeInteraction:
        return None
    p = layer.rejector
    path = "pair_representation.interactions.0"
    installed = type(p) is _InferencePairProduct
    prototype = next(layer.parameters(), None)
    supported = (
        backend == "cuda"
        and task == "inference"
        and prototype is not None
        and prototype.is_cuda
        and prototype.dtype == torch.float32
        and not torch.version.hip
    )
    if not installed and (automatic or not supported):
        return None
    if custom_forward(host) or custom_forward(pair, recursive=True):
        if installed and not supported:
            raise ValueError(
                "Cannot retarget a customized HIP pair; use the original host."
            )
        return None
    if installed and (not supported or not p._unchanged_tables()):
        return Region(
            path,
            "native",
            "Restore the host e3nn pair tensor product.",
            _restore_pair,
            "regional",
            "e3nn",
            "readout_fn",
        )
    if not installed and (
        type(p) is not UvuTensorProduct
        or layer.num_channel != 64
        or (layer.irreps_hidden, layer.irreps_sh, layer.irreps_out)
        != (p.irreps_in1, p.irreps_in2, p.irreps_out)
        or not hasattr(
            getattr(edge_inter, "_RowwisePairProduct", None), "_setup_rowwise"
        )
        or not _ordinary_pair_paths(p)
    ):
        return None
    return Region(
        path,
        "cuda",
        "Row-wise CG on eager FP32 CUDA pair rows in eval mode; otherwise the host tensor product.",
        None if installed else _install_pair,
        "already_regional" if installed else "regional",
        "eqx_cuda",
        "readout_fn",
    )


def _active_slots(host):
    targets = set(host.target_property)
    for target, slots in (
        ("energy", ("energy_readouts", "one_body_magmoms_readout")),
        ("direct_hessian", ("direct_hessian_readout",)),
        ("direct_forces", ("direct_forces_readouts",)),
        ("direct_dipole", ("dipole_readouts",)),
        (
            "direct_polarizability",
            ("direct_polarizability_readout0s", "direct_polarizability_readout2s"),
        ),
        ("abs_final_collinear_magmoms", ("abs_final_collinear_magmoms_readouts",)),
    ):
        if target in targets:
            for name in slots:
                yield target, name
    if targets & {"direct_virials", "direct_stress"}:
        for name in ("direct_virials_readout0s", "direct_virials_readout2s"):
            yield "direct stress/virials", name
    if "charges" in targets:
        slots = {
            "lagrangian": ("chi_readouts", "eta_readouts"),
            "uniform_distribution": ("charges_readouts",),
        }.get(getattr(host, "predict_charges_method", None), ())
        for name in slots:
            yield "charges", name
    # Requesting a schedule does not toggle the module's train/eval mode.
    if host.training:
        yield "DeNS noise", "dens_noise_readouts"
    yield "LES", "les"


def property_regions(host):
    """Report the callable heads of the standard TACE forward, not dormant slots."""
    if not _standard_readout(host):
        return
    for target, name in _active_slots(host):
        head = getattr(host, name, None)
        if not isinstance(head, torch.nn.Module):
            continue
        # A ModuleList is storage; it is the individual heads that are called.
        paths = (
            (f"{name}.{i}" for i in range(len(head)))
            if isinstance(head, torch.nn.ModuleList)
            else (name,)
        )
        for path in paths:
            yield _host(
                path, f"The host owns the {target} readout and its internal dispatch."
            )
    pair = getattr(host, "pair_representation", None)
    if "direct_hessian" not in host.target_property or not isinstance(
        pair, torch.nn.Module
    ):
        return
    yield _host(
        "pair_representation",
        "The host builds directed HIP edge features from geometry and node descriptors.",
    )
    if (
        type(pair) is PairRepresentation
        and not custom_forward(pair)
        and len(pair.interactions)
    ):
        # PairRepresentation.forward consumes interactions[0], not the node stack.
        reason, provider = _pair_interaction(pair.interactions[0])
        yield _host("pair_representation.interactions.0", reason, provider)
