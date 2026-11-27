"""Local execution rules for ordinary spherical TACE ACE products.

Only immutable angular programs are prepared. Coefficients, gates, species
and residuals stay live; no rule needs an architecture name or fixed N.
"""

from functools import partial
from types import SimpleNamespace

import torch
from e3nn import o3

from equivariantx.o3 import (
    ElementLinear,
    MoEElementLinear,
    TransposeIrrepsLayout,
    UuuFullyConnectedSymmetricContraction,
)
from equivariantx.o3.cgtp import TensorProduct as EqxTensorProduct
from equivariantx.o3.generalized import CouplingBlock, GeneralizedContraction
from equivariantx.ops.common.channel_product import channel_product
from equivariantx.ops.common.torch.spherical_product import SphericalProduct
from equivariantx.ops.precision import cpu_matmul_allowed
from tace.models.linear import (
    e3nnElementLinear,
    e3nnLinear,
    e3nnMoEElementLinear,
    has_lora,
    torchLinear,
)
from tace.models.mlp import ScaledSiLU
from tace.models.spherical.fused import UuuSymmetricContraction, UuuTensorProduct
from tace.models.spherical.prod import CgtpACE, MaceACE, NonlinearCgtpACE, TaceACE
from tace.models.spherical.tece_oam_rra import BilinearMoEACE
from tace.ops.bilinear import BilinearContraction
from tace.utils.env import acceleration_enabled

from .matching import custom_forward
from .nonlinear_ace import build_product


class _ScheduledPathGates(TaceACE):
    """Reuse the host's optional gated polynomial without enabling it globally."""

    def contract(self, x):
        if (
            not x.is_cuda
            or not _supported_precision(x)
            or torch.compiler.is_compiling()
            or has_lora(self.linear_up)
            or has_lora(self.linear)
        ):
            return super().contract(x)
        from .nonlinear_ace import run

        # The host does input reordering, stochastic depth and the residual.
        # Keep its eqx_metadata flag absent: a global environment switch must
        # not bypass this guard. Resolve both live linear maps on every call.
        metadata, bindings = self._foundation_path_program
        return run(
            SimpleNamespace(
                eqx_metadata=metadata,
                eqx_bindings=bindings,
                linear_up=self.linear_up,
                linear=self.linear,
            ),
            x,
        )


def _install_path_gates(candidate):
    from .nonlinear_ace import build_path_gates

    candidate._foundation_path_program = build_path_gates(candidate)
    candidate.__class__ = _ScheduledPathGates
    return candidate


def _restore_path_gates(candidate):
    del candidate._foundation_path_program
    candidate.__class__ = TaceACE
    return candidate


def _path_gate_linear(layer):
    if (
        type(layer) is not e3nnLinear
        or type(layer.linear) is not o3.Linear
        or has_lora(layer)
    ):
        return False
    if layer.bias is None:
        return True
    # The host uses these buffers at runtime; the fused expression places
    # scalar biases directly. Never discard a changed bias placement or dtype.
    mask = torch.zeros(layer.irreps_out.dim, dtype=torch.bool, device="cpu")
    index = torch.zeros(layer.irreps_out.dim, dtype=torch.long, device="cpu")
    offset = 0
    for (mul, ir), section in zip(
        layer.irreps_out, layer.irreps_out.slices(), strict=True
    ):
        if ir.is_scalar():
            mask[section] = True
            index[section] = torch.arange(offset, offset + mul, device="cpu")
            offset += mul
    return (
        offset == layer.bias.numel()
        and layer._bias_mask.dtype == layer.bias.dtype
        and not layer._bias_mask.requires_grad
        and torch.equal(layer._bias_mask.detach().cpu(), mask)
        and torch.equal(layer._bias_index.detach().cpu(), index)
    )


def _path_gate_rule(module, *, backend, automatic):
    if custom_forward(module, recursive=True) or "contract" in vars(module):
        return None, "preserved a customized TaceACE product"
    if type(module) is _ScheduledPathGates:
        if backend == "torch":
            if any(b.backend != "torch" for b in module.generalized.blocks):
                return (
                    None,
                    "cannot restore a modified generalized-contraction provider",
                )
            return (
                _restore_path_gates,
                "restore the original Torch path-gated contraction",
            )
        return None, "already uses the scheduled CUDA path-gated contraction"
    if hasattr(module, "eqx_metadata") or hasattr(module, "eqx_bindings"):
        return None, "preserved the host's existing TaceACE path-gate program"
    if backend != "cuda" or automatic:
        return (
            None,
            "retain the native TaceACE program; explicit CUDA opts into path-gate fusion",
        )
    op = getattr(module, "generalized", None)
    if not (
        module.use_generalized_cg
        and not module.use_time_reversal
        and not module.fully_connected
        and module.correlation >= 2
        and type(op) is GeneralizedContraction
        and op.codegen_supported
        and not op.shared_inputs
        and op.gated
        and "packed" not in vars(op)
        and len(op.blocks) + 1 == op.num_factors == module.correlation
        and all(
            type(b) is CouplingBlock
            and b.backend == "torch"
            and "angular_rows" not in vars(b)
            for b in op.blocks
        )
        and type(module.activation) is ScaledSiLU
        and not module.activation.inplace
        and all(_path_gate_linear(m) for m in (module.linear_up, module.linear))
        and all(
            p.is_cuda and p.dtype in (torch.float32, torch.float64)
            for p in module.parameters()
        )
    ):
        return (
            None,
            "requires ordinary independent factors, Torch generalized CG and scalar path gates",
        )
    return (
        _install_path_gates,
        "fused existing TaceACE projections, independent factors and path gates",
    )


def _supported_precision(x):
    return (
        x.device.type in ("cpu", "cuda")
        and x.dtype in (torch.float32, torch.float64)
        and not torch.is_autocast_enabled(x.device.type)
        and (x.is_cuda or x.dtype == torch.float64 or cpu_matmul_allowed())
    )


class _ScheduledCgtpACE(CgtpACE):
    """Run all correlation orders with their original, external coefficient maps."""

    def forward(self, node_feats, node_type, sc, batch):
        if not _supported_precision(node_feats) or (
            not self.agnostic and node_type.dtype != torch.int64
        ):
            return super().forward(node_feats, node_type, sc, batch)
        x = self.linear_up(node_feats)
        elements = 1 if self.agnostic else self.num_elements
        types = (
            x.new_zeros(x.shape[0], dtype=torch.long) if self.agnostic else node_type
        )
        weights = [
            torch.cat([p.reshape(elements, -1) for p in coef.weight], dim=1)
            if isinstance(coef.weight, torch.nn.ParameterList)
            else coef.weight.reshape(elements, -1)
            for coef in self.coefs
        ]
        out = self.scheduled_contraction(x, weights, types)
        for coef in self.coefs:
            if coef.bias is not None:
                bias = (
                    coef.bias
                    if self.agnostic
                    else coef.bias.reshape(elements, -1)[types]
                )
                out = out + bias.index_select(-1, coef._bias_index) * coef._bias_mask
        out = self.linear(out)
        if hasattr(self, "stochastic_depth"):
            out = self.stochastic_depth(out, batch)
        return out if sc is None else torch.add(sc, out, alpha=self.residual_scale)


class _ScheduledUuu(UuuTensorProduct):
    """Unweighted, two-factor tensor product (the factors need not coincide)."""

    def forward(self, x, y, ws=None):
        if ws is not None or not _supported_precision(x):
            return super().forward(x, y, ws)
        if self.schedule_backend == "cuda" and x.is_cuda:
            return channel_product(
                self.schedule_metadata,
                "(((0, 1, 2), 2, 0, 0),)",
                [x, y, x.new_empty((0, self.irreps_out.dim))],
            )[0]
        if self.schedule_backend == "torch":
            return self.scheduled_coupling(x, y, self.tp.weight)
        # A CUDA schedule can be moved to CPU after lowering.
        return super().forward(x, y)


def spherical_bucket_product(module):
    """Installed deterministic node-local programs, never eligibility or a label."""
    if type(module) is _ScheduledBilinearMoEACE:
        op = module.scheduled_contraction
        valid, _ = _bilinear_rule(module, backend="cuda", automatic=False)
        return (
            valid is not None
            and type(op) is BilinearContraction
            and op.backend == "cuda"
            and op.tensor_products[0] is module.aces[0].tp
            and not any(k in vars(module) for k in ("_contract", "_linear_up_features"))
            and ((op.shared is None) == (not hasattr(module, "shared_coefs")))
            and (
                op.shared is None
                or (
                    type(op.shared) is BilinearContraction
                    and op.shared.backend == "cuda"
                    and op.shared.tensor_products[0] is module.aces[0].tp
                )
            )
        )
    if type(module) is _ScheduledCgtpACE:
        op = module.scheduled_contraction
        return (
            type(op) is UuuFullyConnectedSymmetricContraction
            and op.backend == "cuda"
            and len(op.weight_numels) == module.correlation
        )
    if type(module) not in (MaceACE, NonlinearCgtpACE) or module.use_time_reversal:
        return False

    def local_linear(op):
        return (
            type(op) is e3nnLinear and type(op.linear) is o3.Linear and not has_lora(op)
        )

    if not local_linear(module.linear):
        return False
    if type(module) is MaceACE:
        # At node i only its own feature and its species' shared polynomial
        # weights are read. A masked extra node contributes no weight adjoint.
        op = module.product
        return (
            type(op) is UuuSymmetricContraction
            and op.backend == "cuda"
            and not op.use_cue
            and not hasattr(op, "fused_tp")
            and op.has_terms
            and op.layout == "mul_ir"
            and type(module.reshape) is TransposeIrrepsLayout
            and (
                type(module.linear_up) is torch.nn.Identity
                or local_linear(module.linear_up)
            )
            and module.weight.is_cuda
            and module.weight.dtype == torch.float32
        )
    return node_local_path_gates(module, local_linear) and all(
        type(ace) is _ScheduledUuu
        and ace.schedule_backend == "cuda"
        and type(ace.tp) in (o3.TensorProduct, EqxTensorProduct)
        for ace in module.aces
    )


def node_local_path_gates(module, local_linear):
    """Whether the nonlinear ACE coefficients and channel maps act per node."""
    return (
        not module.use_generalized_cg
        and not module.use_time_reversal
        and not hasattr(module, "eqx_metadata")
        and not hasattr(module, "eqx_products")
        and local_linear(module.linear)
        and local_linear(module.linear_up)
        and type(module.embedding) is torch.nn.Embedding
        and not module.embedding.scale_grad_by_freq
        and module.embedding.max_norm is None
        and type(module.gate) is torch.nn.Sequential
        and len(module.gate) == 2
        and type(module.gate[0]) is torchLinear
        and not has_lora(module.gate[0])
        and type(module.gate[1]) is ScaledSiLU
        and len(module.coefs) == module.correlation == len(module.aces) + 1
        and all(local_linear(c) for c in module.coefs)
        and bool(module.aces)
    )


def _lower_product(candidate, *, backend):
    ref = next(candidate.parameters())
    if type(candidate) in (CgtpACE, _ScheduledCgtpACE):
        schedule = UuuFullyConnectedSymmetricContraction(
            [ace.tp for ace in candidate.aces],
            [coef.linear for coef in candidate.coefs],
            backend=backend,
        )
        # A schedule owns only regenerable tables, not canonical model state.
        for child in schedule.modules():
            child._non_persistent_buffers_set.update(child._buffers)
        candidate.scheduled_contraction = schedule.to(ref).train(candidate.training)
        candidate.__class__ = _ScheduledCgtpACE
    else:
        # Native forward normally selects its globally configured fusion when
        # these optional string programs exist. On this private copy select our
        # local product stages while retaining the exact original gates/maps.
        for name in ("eqx_metadata", "eqx_products"):
            if hasattr(candidate, name):
                delattr(candidate, name)
        for ace in candidate.aces:
            tp = ace.tp
            if backend == "torch":
                ace.scheduled_coupling = (
                    SphericalProduct(
                        (tp.irreps_in1, tp.irreps_in2, tp.irreps_out),
                        tp.instructions,
                        backend="torch",
                    )
                    .to(ref)
                    .train(candidate.training)
                )
            else:
                ace.schedule_metadata = build_product(tp)
            ace.schedule_backend = backend
            ace.__class__ = _ScheduledUuu
    return candidate


def _select_mace_backend(candidate, *, backend):
    # Both providers consume the *same* prepared contraction and external
    # Parameter. Do not regenerate U matrices, replace forward, or repack weights.
    candidate.product.backend = backend
    candidate.product.use_eqx = backend == "cuda"
    return candidate


class _ScheduledBilinearMoEACE(BilinearMoEACE):
    """Fuse only the second-order coefficient term; keep all weights external."""

    def _contract(self, features, base, gates, weight, types, shared_weight):
        # The mixed adjoints allocate outputs with one row for every input
        # coefficient row. For a small batch taken from a large species table,
        # route one live row per node instead. index_select's adjoint adds
        # repeated species back to the same parameter, including higher grads.
        # Do not extract unique types (that would synchronize CUDA); bound the
        # additional forward gather to at most one quarter of the original table.
        if (
            self.training
            and torch.is_grad_enabled()
            and weight.requires_grad
            and 0 < types.numel() * 4 <= weight.shape[0]
        ):
            weight = weight.index_select(0, types)
            types = torch.arange(types.numel(), device=types.device, dtype=types.dtype)
        return self.scheduled_contraction(
            features, base, gates, weight, types, shared_weight
        )

    def forward(self, node_feats, node_type, sc, batch):
        if (
            not node_feats.is_cuda
            or not _supported_precision(node_feats)
            or torch.compiler.is_compiling()
            or (not self.agnostic and node_type.dtype != torch.int64)
            or has_lora(self.coefs[1])
            or (hasattr(self, "shared_coefs") and has_lora(self.shared_coefs[1]))
        ):
            # The original module has no eqx_ace: its complete fallback is intact.
            return super().forward(node_feats, node_type, sc, batch)
        x, base, gates = self._linear_up_features(node_feats)
        out = (
            self.coefs[0](x) if self.agnostic else self.coefs[0](x, node_type=node_type)
        )
        shared = self.shared_coefs[0](x) if hasattr(self, "shared_coefs") else None
        out = out + self._bilinear_coefficients(
            self._contract,
            self.coefs[1],
            x,
            base,
            gates[:, self._ace_gate_slices[0]],
            node_type,
        )
        if shared is not None:
            out = self._merge_shared_expert(out, shared)
        out = self.linear(out)
        if hasattr(self, "stochastic_depth"):
            out = self.stochastic_depth(out, batch)
        return out if sc is None else torch.add(sc, out, alpha=self.residual_scale)


def _install_bilinear(module):
    coef = module.coefs[1]
    module.scheduled_contraction = BilinearContraction(
        module.aces[0].tp,
        coef.linear,
        getattr(coef, "num_experts", 1),
        backend="cuda",
        shared_linear=module.shared_coefs[1].linear
        if hasattr(module, "shared_coefs")
        else None,
    ).train(module.training)
    module.__class__ = _ScheduledBilinearMoEACE
    return module


def _bilinear_rule(module, *, backend, automatic):
    if hasattr(module, "eqx_ace"):
        return None, "preserved the host's existing bilinear contraction"
    if backend != "cuda" or automatic:
        return None, "explicit CUDA opts into second-order bilinear coefficient fusion"
    if not (
        module.use_bilinear_gate
        and module.correlation == 2
        and not module.use_time_reversal
        and len(module.aces) == 1
        and len(module.coefs) == 2
        and not custom_forward(module, recursive=True)
        and "_bilinear_coefficients" not in vars(module)
        and "_merge_shared_expert" not in vars(module)
        and all(
            p.is_cuda and p.dtype in (torch.float32, torch.float64)
            for p in module.parameters()
        )
    ):
        return None, "requires an ordinary CUDA FP32/FP64 gated second-order product"
    ace = module.aces[0]
    if (
        type(ace) is not UuuTensorProduct
        or type(ace.tp) not in (o3.TensorProduct, EqxTensorProduct)
        or ace.tp.internal_weights
        or not ace.tp.instructions
        or any(
            p.connection_mode != "uuu" or not p.has_weight for p in ace.tp.instructions
        )
    ):
        return None, "requires the host's externally gated spherical uuu product"
    coef = module.coefs[1]
    expected = (
        e3nnLinear
        if module.agnostic
        else (e3nnMoEElementLinear if module.num_expert > 1 else e3nnElementLinear)
    )
    shared = getattr(module, "shared_coefs", ())
    if (
        type(coef) is not expected
        or coef.irreps_in != ace.tp.irreps_out.simplify()
        or (shared and (len(shared) != 2 or type(shared[1]) is not e3nnLinear))
        or any(
            type(m.linear) is not o3.Linear or m.linear.internal_weights or has_lora(m)
            for m in (coef, *shared[1:])
        )
    ):
        return None, "preserved customized or LoRA expert coefficient maps"
    if not module.agnostic:
        implementation = (
            MoEElementLinear if expected is e3nnMoEElementLinear else ElementLinear
        )
        if any(
            op is not None
            and (
                type(op) is not implementation
                or op.element_mode != "single"
                or op.layout != "mul_ir"
                or op.projection is not None
            )
            for op in (coef.eqx_linear, coef.torch_linear)
        ):
            return None, "preserved a custom routed coefficient implementation"
    return (
        _install_bilinear,
        "EQX gated bilinear contraction with live expert and shared coefficient maps",
    )


def host_product_provider(module):
    """Identify the unchanged host contraction, not all operations in the block."""
    if type(module) not in (
        MaceACE,
        CgtpACE,
        TaceACE,
        BilinearMoEACE,
    ) or custom_forward(module, recursive=True):
        return None
    if next(module.parameters()).dtype not in (torch.float32, torch.float64):
        return None
    if type(module) is BilinearMoEACE:
        op = getattr(module, "eqx_ace", None)
        if (
            type(op) is BilinearContraction
            and next(module.parameters()).is_cuda
            and op.backend in ("cuda", "torch")
            and not has_lora(module.coefs[1])
            and not (
                hasattr(module, "shared_coefs") and has_lora(module.shared_coefs[1])
            )
        ):
            return f"eqx_{op.backend}"
        return None
    if type(module) is TaceACE:
        op = getattr(module, "generalized", None)
        if (
            not module.use_generalized_cg
            or type(op) is not GeneralizedContraction
            or not op.blocks
            or "packed" in vars(op)
            or "contract" in vars(module)
            or not all(type(b) is CouplingBlock for b in op.blocks)
        ):
            return None
        p = next(module.parameters())
        if p.device.type not in ("cpu", "cuda"):
            return None
        if (
            p.is_cuda
            and hasattr(module, "eqx_metadata")
            and acceleration_enabled("eqx", kernel="product")
            and not (module.training and acceleration_enabled("compile"))
            and not any(has_lora(m) for m in (module.linear_up, module.linear))
        ):
            return "eqx_cuda"
        backend = op.blocks[0].backend if p.is_cuda else "torch"
        if backend in ("torch", "cuda") and (
            not p.is_cuda or all(b.backend == backend for b in op.blocks)
        ):
            return f"eqx_{backend}"
        return None
    if type(module) is MaceACE:
        op = module.product
        if type(op) is not UuuSymmetricContraction:
            return None
        device = module.weight.device.type
        if device == "cpu":
            # The host skips CuEq as well as its native CUDA kernel on CPU.
            return "eqx_torch"
        if device != "cuda":
            return None
        if op.use_cue and op.has_terms:
            from tace.models.backends.cuequivariance import CueUuuSymmetricContraction

            return (
                "cueq"
                if type(getattr(op, "fused_tp", None)) is CueUuuSymmetricContraction
                else None
            )
    else:
        op = getattr(module, "eqx_ace", None)
        if (
            type(op) is not UuuFullyConnectedSymmetricContraction
            or module.use_generalized_cg
            or any(has_lora(coef) for coef in module.coefs)
        ):
            return None
    if next(module.parameters()).device.type == "cpu":
        return "eqx_torch"
    return f"eqx_{op.backend}" if op.backend in ("cuda", "torch") else None


def product_rule(module, *, backend, task, automatic=False):
    """Return a lowering on an isolated copy or a reason to leave the block native."""
    if backend not in ("torch", "cuda"):
        return None, "product provider must be torch or cuda"
    if task not in ("inference", "training"):
        return None, "product rule requires ordinary inference or training"
    if type(module) in (TaceACE, _ScheduledPathGates):
        return _path_gate_rule(module, backend=backend, automatic=automatic)
    if type(module) is BilinearMoEACE:
        return _bilinear_rule(module, backend=backend, automatic=automatic)
    if type(module) is MaceACE:
        contraction = module.product
        if type(contraction) is not UuuSymmetricContraction:
            return None, "preserved a custom MaceACE contraction"
        if contraction.use_cue or hasattr(contraction, "fused_tp"):
            return None, "preserved the existing CuEq MaceACE provider"
        if contraction.backend == backend:
            return (
                None,
                f"MaceACE already uses the host's {backend} symmetric contraction",
            )
        if (
            module.weight.dtype not in (torch.float32, torch.float64)
            or module.use_time_reversal
            or custom_forward(module, recursive=True)
        ):
            return None, "preserved unsupported or customized MaceACE execution"
        return (
            partial(_select_mace_backend, backend=backend),
            f"select {backend} for the existing EQX symmetric contraction; keep live MaceACE weights",
        )
    if type(module) is _ScheduledCgtpACE:
        if (
            type(module.scheduled_contraction)
            is not UuuFullyConnectedSymmetricContraction
        ):
            raise ValueError(
                "Cannot replace a customized symmetric-contraction schedule; use the original host model."
            )
        if module.scheduled_contraction.backend == backend:
            return None, f"already uses the scheduled {backend} symmetric contraction"
        if (
            custom_forward(module.scheduled_contraction, recursive=True)
            or module.scheduled_contraction.state_dict()
        ):
            raise ValueError(
                "Cannot replace a customized symmetric-contraction schedule; use the original host model."
            )
        return partial(
            _lower_product, backend=backend
        ), f"retarget live symmetric contraction to {backend}"
    if type(module) not in (CgtpACE, NonlinearCgtpACE):
        return None, "no local rule for this product type"
    ref = next(module.parameters())
    if ref.dtype not in (torch.float32, torch.float64):
        return None, "product rule requires FP32 or FP64"
    if module.use_time_reversal or module.use_generalized_cg:
        return None, "time-reversal and generalized CG products retain native execution"
    if not module.aces or len(module.coefs) != len(module.aces) + 1:
        return None, "expected at least one recursive product with a map per order"
    if any(has_lora(m) for m in module.modules()):
        return None, "LoRA product weights retain native execution"
    if getattr(module, "eqx_ace", None) is not None or (
        ref.is_cuda
        and hasattr(module, "eqx_metadata")
        and acceleration_enabled("eqx", kernel="product")
    ):
        return None, "preserved the host's existing product schedule"
    if automatic and backend == "torch" and type(module) is NonlinearCgtpACE:
        return (
            None,
            "prefer the native nonlinear Torch schedule; explicit torch enables local CG lowering",
        )
    if custom_forward(module, recursive=True):
        return None, "preserved customized or hooked product execution"
    for ace in module.aces:
        if type(ace) is not UuuTensorProduct or type(ace.tp) not in (
            o3.TensorProduct,
            EqxTensorProduct,
        ):
            return None, "expected native spherical tensor-product factors"
        tp = ace.tp
        if not tp.instructions or any(
            ins.connection_mode != "uuu" or ins.has_weight for ins in tp.instructions
        ):
            return None, "only nonempty unweighted uuu couplings are supported"
        widths = {
            m
            for irreps in (tp.irreps_in1, tp.irreps_in2, tp.irreps_out)
            for m, _ in irreps
        }
        if len(widths) != 1:
            return None, "product channel multiplicities must agree"
    standard = type(module) is CgtpACE
    coef_type = e3nnElementLinear if standard and not module.agnostic else e3nnLinear
    if any(
        type(coef) is not coef_type or type(coef.linear) is not o3.Linear
        for coef in module.coefs
    ):
        return None, "custom coefficient maps retain native execution"
    reason = (
        "EQX symmetric contraction and live coefficient maps"
        if standard
        else "EQX channel products with native invariant gates"
    )
    return partial(_lower_product, backend=backend), reason
