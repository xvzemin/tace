"""Reciprocal execution for degree-wise CGTP radial networks."""

from copy import deepcopy

import pytest
import torch

from equivariantx_foundation import optimize
from equivariantx_foundation.models.tace import plan
from equivariantx_foundation.models.tace.interactions import grouped_radial_geometry
from equivariantx_foundation.tests.test_tace_plan import graph
from tace.models.adapter import TensorModel
from tace.models.spherical.default import DEFAULT_MODEL_CONFIG
from tace.models.spherical.tace import SphericalTACE
from tace.utils.env import ACCELERATION_ENV


def model(monkeypatch, device, basis):
    for name in ACCELERATION_ENV.values():
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    torch.set_default_dtype(torch.float32)
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg.update(
        num_layers=2,
        num_channel=8,
        lmax=2,
        mmax=2,
        Lmax=2,
        cutoff=4.0,
        statistics=[
            dict(
                atomic_numbers=[1, 8],
                atomic_energy={1: 0.0, 8: 0.0},
                avg_num_neighbors=3.0,
            )
        ],
        target_property=["energy", "forces", "stress"],
    )
    cfg["edge_update"]["type"] = "identity"
    cfg["edge_embedding"]["type"] = "identity"
    cfg["radial_basis"].update(
        radial_basis=basis, num_radial_basis=[4, 3, 2], hidden=[16, 32], mlp_norm=None
    )
    cfg["scale_shift"]["enable"] = False
    return TensorModel(SphericalTACE(**cfg)).to(device).train()


@pytest.mark.parametrize("basis", ["jn", "ejn", "spb"])
def test_grouped_radial_requires_all_branches_and_live_path_layout(monkeypatch, basis):
    if basis == "spb":
        pytest.importorskip("physical_basis")
    rep = model(monkeypatch, "cpu", basis).readout_fn.representation
    assert grouped_radial_geometry(rep)
    network = rep.interactions[-1].edge_info
    with network.branches[0].register_forward_hook(lambda *args: None):
        assert not grouped_radial_geometry(rep)
    assert grouped_radial_geometry(rep)
    order = torch.arange(network.out_features)
    network.output_order = order.flip(0)
    network.inverse_order = order.flip(0)
    assert not grouped_radial_geometry(rep)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip, reason="CUDA regional j_n"
)
@pytest.mark.parametrize("basis", ["jn", "ejn", "spb"])
def test_streamed_radial_region_keeps_complete_efs_gradients(monkeypatch, basis):
    if basis == "spb":
        pytest.importorskip("physical_basis")
    torch.manual_seed(43)
    original = model(monkeypatch, "cuda", basis)

    def uses_jn(**kw):
        return any(
            "degree-wise" in r.reason
            for r in plan(original, regional_only=True, **kw).regions
        )

    assert not uses_jn()
    assert uses_jn(backend="cuda", task="inference") == (basis != "jn")
    with monkeypatch.context() as patch:
        patch.setenv("TACE_MEMORY_FIRST", "0")
        assert not uses_jn(backend="cuda", task="training")
    assert uses_jn(backend="cuda", task="training")
    adapted = optimize(
        original, family="tace", task="training", backend="cuda", regional_only=True
    )
    assert adapted.uses_reciprocal_weights() and not original.uses_reciprocal_weights()
    assert adapted.state_dict().keys() == original.state_dict().keys()
    pp = dict(original.named_parameters())
    assert all(p is pp[k] for k, p in adapted.named_parameters())
    data = graph("cuda")
    assert (
        adapted.prepare_property_data(data)["edge_pair_rows"].numel()
        == data["edge_index"].size(1) // 2
    )

    def step(m):
        out = m(data)
        pred = tuple(out[k] for k in ("energy", "forces", "stress"))
        objective = sum(x.square().mean() for x in pred)
        grad = torch.autograd.grad(objective, tuple(pp.values()), allow_unused=True)
        return pred, grad

    expected, grad0 = step(original)
    actual, grad1 = step(adapted)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=5e-5)
    for x, y in zip(grad1, grad0, strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, atol=5e-6, rtol=3e-4)
    other = optimize(
        adapted, family="tace", task="training", backend="torch", regional_only=True
    )
    assert not other.uses_reciprocal_weights()
