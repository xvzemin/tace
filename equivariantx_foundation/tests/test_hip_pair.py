"""The explicit HIP inference rule preserves native training, state and row gradients."""

from pathlib import Path

import pytest
import torch
from e3nn import o3
from omegaconf import OmegaConf
from torch.utils._python_dispatch import TorchDispatchMode

from equivariantx_foundation.models.tace.heads import _ordinary_pair_paths
from equivariantx_foundation.models.tace.planner import optimize_model, plan
from tace.models.adapter import TensorModel
from tace.models.spherical.fused import UvuTensorProduct
from tace.models.spherical.paths import generate_paths
from tace.models.spherical.tace import SphericalTACE
from tace.utils.env import ACCELERATION_ENV


def test_hip_pair_path_match_checks_actual_instructions_and_cg(monkeypatch):
    monkeypatch.setenv("TACE_USE_OEQ", "0")
    x, sh = o3.Irreps("64x0e+64x1o+64x1e+64x2e"), o3.Irreps.spherical_harmonics(3)
    paths, out = generate_paths(o3.Irreps("0e+1o+1e+2e"), x, sh)
    product = UvuTensorProduct(x, sh, out, paths, shared_weights=False)
    assert _ordinary_pair_paths(product)
    tp = product.tp
    with monkeypatch.context() as mp:
        mp.setattr(tp, "instructions", list(reversed(tp.instructions)))
        assert not _ordinary_pair_paths(product)
    cg = next(b for n, b in tp.named_buffers() if "_w3j_" in n)
    cg.neg_()
    assert not _ordinary_pair_paths(product)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP CUDA pair selection")
def test_explicit_hip_inference_pair_dispatch_and_live_state(monkeypatch):
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    recipes = Path(__file__).resolve().parents[2] / "example" / "train"
    cfg = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(recipes / "TACE/tace.yaml").model.config,
            OmegaConf.load(recipes / "HIP-TACE/hip-tace.yaml").model.config,
        ),
        resolve=False,
    )
    del cfg["_target_"], cfg["wrapper"]
    cfg.update(
        atomic_numbers=[1],
        statistics=[
            dict(
                atomic_numbers=[1],
                atomic_energy={1: 0.0},
                rms_forces={1: 1.0},
                avg_num_neighbors=4.0,
            )
        ],
        target_property=["energy", "forces", "direct_hessian"],
    )
    model = TensorModel(SphericalTACE(**cfg)).cuda().float().eval()
    path = "readout_fn.pair_representation.interactions.0"

    def region(m, backend="cuda", task="inference"):
        return next(
            r
            for r in plan(m, backend=backend, task=task, regional_only=True).regions
            if r.module_path == path
        )

    def optimize(m, backend="cuda", task="inference"):
        return optimize_model(m, backend=backend, task=task, regional_only=True)

    native = model.get_submodule(path).rejector
    assert type(native) is UvuTensorProduct
    assert (
        region(model, None).provider
        == region(model, "cuda", "training").provider
        == "e3nn"
    )
    with native.tp.register_forward_hook(lambda *_: None):
        assert region(model).lower is None
    cg_name = next(n for n, b in native.tp.named_buffers() if "_w3j_" in n)
    cg = native.tp.get_buffer(cg_name)
    with torch.no_grad():
        cg.neg_()
    assert region(model).lower is None
    with torch.no_grad():
        cg.neg_()
    assert region(model).provenance == "regional"
    fast = optimize(model)
    assert fast.state_dict().keys() == model.state_dict().keys()
    assert all(
        a is b for a, b in zip(fast.parameters(), model.parameters(), strict=True)
    )
    assert region(fast).provenance == "already_regional"
    p = fast.get_submodule(path).rejector

    def rnd(d):
        return torch.randn(13, d, device="cuda", requires_grad=True)

    x, y, w = (
        rnd(native.irreps_in1.dim),
        rnd(native.irreps_in2.dim),
        rnd(native.weight_numel),
    )

    def vals(op):
        out = op(x, y, w)
        g = torch.autograd.grad(out.sin().mean(), (x, y, w), create_graph=True)
        h = torch.autograd.grad(g[0].sin().sum(), (y, w))
        return out, g, h

    expected = vals(native)

    class Seen(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.hits = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.hits += func == torch.ops.eqx.o3_contraction.default
            return func(*args, **(kwargs or {}))

    # The actual EQX custom op, not a planner receipt or a wrapper hook.
    with Seen() as seen:
        out = p(x, y, w)
    assert seen.hits == 1
    torch.testing.assert_close(vals(p), expected, rtol=4e-4, atol=1e-5)
    with p.tp.register_forward_hook(lambda *_: None), Seen() as seen:
        torch.testing.assert_close(p(x, y, w), expected[0])
    assert seen.hits == 0
    fast.train()
    with Seen() as seen:
        torch.testing.assert_close(p(x, y, w), expected[0])
    assert seen.hits == 0
    fast.eval()
    for backend, task in (("torch", "inference"), ("cuda", "training")):
        restored = optimize(fast, backend, task)
        assert type(restored.get_submodule(path).rejector) is UvuTensorProduct
        assert region(restored, backend, task).provider == "e3nn"
    # A customized trainable CG coefficient must keep contributing to the result.
    live = p.tp.get_buffer(cg_name).requires_grad_()
    with Seen() as seen:
        grad = torch.autograd.grad(p(x, y, w).square().sum(), live)[0]
    assert seen.hits == 0 and grad.abs().max() > 0
