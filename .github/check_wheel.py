"""Smoke-test the installed TACE wheel outside the source tree."""

import os
import sys
from importlib import import_module
from importlib.metadata import distribution
from pathlib import Path

import torch
from packaging.requirements import Requirement


def check_model(
    device, *, cartesian=False, rotary=False, radial=None, foundation=False
):
    from copy import deepcopy

    from ase import Atoms
    from torch_geometric.data import Batch

    from tace.dataset.graph import from_atoms
    from tace.dataset.quantity import KeySpecification
    from tace.lightning import create_model
    from tace.models.spherical.default import DEFAULT_MODEL_CONFIG

    targets = ["energy", "forces"]
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg.update(num_layers=1, num_channel=2, Lmax=1, lmax=1, mmax=1, cutoff=3.0)
    if cartesian:
        cfg["_target_"] = "tace.models.CartesianTACE"
        cfg["atomic_basis"]["type"] = "co3"
        cfg["product_basis"].update(type="nonlinear_ictd", correlation=2)
    elif rotary:
        cfg.update(num_layers=2, parity=False)
        cfg["atomic_basis"].update(type=["cgtp", "so2"], num_head=1, edge_ace_hidden=2)
        cfg["edge_update"]["type"] = "element2"
    if radial:
        cfg["radial_basis"].update(radial_basis=radial, num_radial_basis=[3, 2])
    cfg["scale_shift"]["scale_type"] = None
    cfg["layer_norm"]["final_norm_type"] = "soft_cap_rms_norm"
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: -1.0}, avg_num_neighbors=1)
    ]
    model = create_model(cfg, statistics, targets, []).to(device).train()
    assert (
        model.num_fidelities
        == model.readout_fn.atomic_energy_layer.atomic_energy.size(0)
        == 1
    )
    graph = from_atoms(
        model.get_torch_element(),
        Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]]),
        model.get_cutoff(),
        max_neighbors=None,
        keyspec=KeySpecification(),
        target_property=targets,
        embedding_property=[],
        training=False,
        neighborlist_backend="ase",
    )
    data = Batch.from_data_list([graph]).to(device).to_dict()
    if foundation:
        from equivariantx_foundation import optimize

        host = model.double()
        model = optimize(host, family="tace", task="training")
        assert model is not host
        params = dict(host.named_parameters())
        assert all(params[n] is p for n, p in model.named_parameters())
        expected = host(data)
    outputs = model(data)
    if foundation:
        for key in targets:
            torch.testing.assert_close(outputs[key], expected[key])
    assert outputs["forces"].shape == (2, 3)
    loss = outputs["energy"].sum() + outputs["forces"].square().sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and torch.isfinite(loss) and all(g.isfinite().all() for g in grads)


def main():
    dist = distribution("tace")
    contents = {p.as_posix() for p in dist.files}
    source = Path(__file__).resolve().parents[1]
    for package in ("equivariantx_foundation", "equivariantx", "tace"):
        path = f"{package}/__init__.py"
        assert path in contents
        module = import_module(package)
        assert Path(module.__file__).resolve() == Path(dist.locate_file(path)).resolve()
        if package != "tace":
            assert "tace" not in sys.modules
        for file in (source / package).rglob("*"):
            rel = file.relative_to(source)
            if not {"tests", "docs", "build", "__pycache__"}.intersection(
                rel.parts
            ) and file.suffix in {".py", ".cu", ".cpp", ".cuh", ".h"}:
                assert rel.as_posix() in contents, rel
    assert not any("/tests/" in p or "/build/" in p for p in contents)
    assert {
        "equivariantx/ops/common/cuda/csrc/runtime.cpp",
        "equivariantx/ops/cuda/co2/frame_matrix.cu",
        "tace/dataset/stats_yaml/omat24.yaml",
    } <= contents
    assert not {
        Requirement(r).name.lower().replace("_", "-") for r in dist.requires
    } & {"equivariantx", "equivariantx-foundation"}

    from equivariantx.o3 import ElementwiseTensorProduct
    from equivariantx.ops import resolve_backend
    from equivariantx_foundation import optimize

    device = "cuda" if torch.version.cuda and torch.cuda.is_available() else "cpu"
    assert resolve_backend() == ("cuda" if device == "cuda" else "torch")
    tp = ElementwiseTensorProduct("1o", "1o", filter_ir_out=["0e"]).to(device)
    vector = torch.tensor([[1.0, 2.0, 3.0]], device=device, requires_grad=True)
    value = tp(vector, vector)
    torch.testing.assert_close(value, vector.square().sum(-1, keepdim=True) / 3**0.5)
    torch.testing.assert_close(
        torch.autograd.grad(value.sum(), vector)[0], 2 * vector / 3**0.5
    )
    try:
        optimize(torch.nn.Linear(1, 1, device="cpu"), family="tace")
    except RuntimeError as error:
        assert "requires an NVIDIA CUDA device" in str(error)
    else:
        raise AssertionError("Foundation accepted CPU execution")

    os.environ["TACE_USE_EQX"] = "0"
    check_model(device)
    check_model(device, radial="ejn")
    if device == "cuda":
        check_model(device, foundation=True)
    os.environ["TACE_USE_EQX"] = "1"
    check_model(device, cartesian=True)
    check_model(device, rotary=True)
    print(
        f"TACE wheel: bundled packages, {device} operators, energy/forces and gradients passed"
    )


if __name__ == "__main__":
    main()
