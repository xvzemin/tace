import json
import os
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union
from unittest.mock import Mock

import pytest
import torch

import tace.models.compile.wrapper as compile_wrapper
from tace.dataset.quantity import (
    SUPPORTED_EMBEDDING_PROPERTIES,
    get_embedding_property,
    get_need_property,
)
from tace.lightning.torch_model import (
    _prune_removed_keys,
    create_model,
)
from tace.models.compile.aot import (
    TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY,
    TACE_AOTI_LUNA_OPS_ENTRY,
    AOTICompiledTensorModel,
    _custom_ops_libs_from_model,
    _embed_custom_ops_libs,
    _ensure_sample_inputs,
    _export_metadata,
    _graph_aoti_input_keys,
    _graph_dynamic_shapes,
    _import_custom_ops_libs,
    _size_oblivious_export,
    _synthetic_graph_sample,
)
from tace.models.compile.compile import compiled_call, trace_and_compile, trace_to_fx
from tace.models.compile.wrapper import CompileTensorModel, _FlatE3nnCompileModel
from tace.models.spherical.default import DEFAULT_MODEL_CONFIG


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_manual_gradient_sync_preserves_dtype_and_views(monkeypatch, backend):
    from types import SimpleNamespace

    from tace.utils.strategy import SimpleDDPStrategy

    model = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype))
            for dtype in (torch.float32, torch.float64, torch.float32)
        ]
    )
    for i, parameter in enumerate(model[:2]):
        parameter.grad = torch.full((3, 2), float(i + 1), dtype=parameter.dtype).T
    gradients = [parameter.grad for parameter in model]
    seen = []

    def reduce(flat, op, async_op):
        assert async_op
        expected = (
            torch.distributed.ReduceOp.SUM
            if backend == "gloo"
            else torch.distributed.ReduceOp.AVG
        )
        assert op == expected
        seen.append(flat.dtype)

        # Complete the simulated collective only when its work is awaited.
        def wait():
            flat.add_(2)
            if backend == "gloo":
                flat.mul_(2)

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(torch.distributed, "get_backend", lambda: backend)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    SimpleDDPStrategy.post_backward(SimpleNamespace(model=model), None)
    assert seen == [torch.float32, torch.float64]
    for i, parameter in enumerate(model[:2]):
        assert parameter.grad is gradients[i]
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter.grad, i + 3)
        )
    assert model[2].grad is None


@pytest.mark.parametrize(
    "backend,device", [("eager", "cpu"), ("inductor", "cpu"), ("inductor", "cuda")]
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_joint_training_gradients(monkeypatch, backend, device, dtype, uncertainty):
    import inspect

    from tace.models.adapter import TensorModel
    from tace.utils.loss.normal import NormalLoss
    from tace.utils.loss.uncertainty import UncertaintyLoss

    if "dynamic_shapes" not in inspect.signature(torch.compile).parameters:
        pytest.skip("Joint training requires explicit dynamic shapes")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_default_dtype(dtype)
    monkeypatch.setenv("TACE_USE_COMPILE", "1")
    monkeypatch.setenv("TACE_USE_EQX", "1")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["product_basis"]["type"] = "nonlinear_cgtp"
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    targets = ["energy", "forces", "stress"]
    config["target_property"] = targets
    stats = [dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)]
    model = create_model(config, stats, targets, [])
    model.readout_fn.compile_backend = backend
    model = model.to(device)
    next(model.parameters()).requires_grad_(False)
    model.readout_fn.register_parameter(
        "unused", torch.nn.Parameter(next(model.parameters()).new_ones(2))
    )
    loss_fn = (UncertaintyLoss if uncertainty else NormalLoss)(
        targets,
        ["mae_energy_per_atom", "l2mae_forces", "mae_voigt_stress"],
        [1.0, 2.0, 3.0],
    )
    loss_fn.to(next(model.parameters()).device)
    parameters = tuple(
        p for p in (*model.parameters(), *loss_fn.parameters()) if p.requires_grad
    )
    data = _synthetic_graph_sample(model)
    data["entropy"] = data["positions"].new_ones(2)
    for key, shape in (("energy", (2,)), ("forces", (4, 3)), ("stress", (2, 3, 3))):
        data[key] = data["positions"].new_full(shape, 0.1)
        data[f"{key}_weight"] = data["entropy"].clone()
    data["stress_weight"][1] = 0
    data["stress"][1] = float("nan")
    accumulated = [None] * len(parameters)
    for step in range(2):
        batch = {key: value.clone() for key, value in data.items()}
        batch["positions"] *= 1 + step * 0.1
        if step:
            batch["edge_index"] = batch["edge_index"][:, :3]
            batch["edge_shifts"] = batch["edge_shifts"][:3]
            batch["stress_weight"].zero_()
        expected = TensorModel.forward(model, dict(batch))
        reference = loss_fn(expected, batch)
        gradients = torch.autograd.grad(reference, parameters, allow_unused=True)
        state = (*model.parameters(), *model.buffers(), *loss_fn.parameters())
        values = [value.detach().clone() for value in state]
        input_values = {key: value.detach().clone() for key, value in batch.items()}
        actual, loss = model.forward_loss(batch, loss_fn)
        current = (*model.parameters(), *model.buffers(), *loss_fn.parameters())
        assert len(current) == len(state)
        for original, value, saved in zip(state, current, values):
            assert original is value
            torch.testing.assert_close(value, saved, atol=0, rtol=0)
        for key, value in input_values.items():
            torch.testing.assert_close(
                batch[key], value, atol=0, rtol=0, equal_nan=True
            )
        tolerance = 2e-4 if dtype == torch.float32 else 2e-9
        torch.testing.assert_close(loss, reference, atol=tolerance, rtol=tolerance)
        for key in targets:
            torch.testing.assert_close(
                actual[key], expected[key], atol=tolerance, rtol=tolerance
            )
        (loss / 2).backward()
        for i, (parameter, grad) in enumerate(zip(parameters, gradients)):
            if grad is None:
                assert parameter.grad is None
            else:
                accumulated[i] = (
                    grad / 2 if accumulated[i] is None else accumulated[i] + grad / 2
                )
                torch.testing.assert_close(
                    parameter.grad, accumulated[i], atol=tolerance, rtol=tolerance
                )
        assert not any(
            value.grad is not None for value in batch.values() if value.is_leaf
        )
        with torch.no_grad():
            for parameter in parameters:
                parameter.mul_(0.99)
    assert len(model._compiled_cache) == 1
    expected_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    for parameter, gradient in zip(expected_parameters, accumulated):
        parameter.grad = gradient
    for values in (parameters, expected_parameters):
        torch.nn.utils.clip_grad_norm_(values, 1.0)
        torch.optim.AdamW(values, lr=1e-3).step()
    for actual, expected in zip(parameters, expected_parameters):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_loss_aliases_and_scaling(device):
    from tace.models.compile.joint import ParameterLoss

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a, b = (torch.nn.Parameter(torch.ones(3, device=device)) for _ in range(2))
    shared = torch.ones(3, device=device)
    loss = ParameterLoss.apply(shared.new_tensor(1.0), (shared, shared), a, b)
    (loss * 4).backward(retain_graph=True)
    assert a.grad.data_ptr() != b.grad.data_ptr()
    a.grad.zero_()
    loss.backward()
    torch.testing.assert_close(a.grad, torch.ones_like(shared))
    torch.testing.assert_close(b.grad, torch.full_like(shared, 5.0))
    torch.testing.assert_close(shared, torch.ones_like(shared))


def test_compiled_shared_state(double_precision):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.linear.register_buffer("scale", torch.randn(3))
            self.linear.weight_alias = self.linear.weight
            self.linear.register_buffer("scale_alias", self.linear.scale)
            self.alias = self.linear
            self.tied = torch.nn.Linear(3, 3)
            self.tied.weight = self.linear.weight
            self.tied.register_buffer("scale", self.linear.scale)

        def forward(self, x):
            return (
                self.linear(x) * self.linear.scale
                + self.alias(x) * self.alias.scale
                + self.tied(x) * self.tied.scale
                + (x @ self.linear.weight_alias.T) * self.linear.scale_alias
            ).sin()

    torch.compiler.reset()
    model = Model()
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    x = torch.randn(4, 3, requires_grad=True)
    compiled, parameters, buffers = trace_and_compile(model, (x,), backend="aot_eager")
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        assert value is state[name]
    for size in (4, 7):
        x = torch.randn(size, 3, requires_grad=True)
        actual = compiled_call(compiled, model, (x,), parameters, buffers)
        expected = model(x)
        torch.testing.assert_close(actual, expected)
        gradients = [
            torch.autograd.grad(value.square().sum(), (x, *model.parameters()))
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        with torch.no_grad():
            model.linear.weight.add_(0.1)
            model.linear.scale.mul_(0.9)


@pytest.mark.parametrize("supports_mixed_reduction", [False, True])
def test_compile_avoids_size_dependent_reduction_fusion(
    monkeypatch, supports_mixed_reduction
):
    from torch._inductor import config

    key = "triton.mix_order_reduction"
    supported = dict(config.get_config_copy())
    if supports_mixed_reduction:
        supported[key] = True
    else:
        supported.pop(key, None)
    monkeypatch.setitem(vars(config), "get_config_copy", lambda: supported)
    received = {}

    def compile_graph(graph, **kwargs):
        received.update(kwargs)
        return graph

    monkeypatch.setattr(torch, "compile", compile_graph)
    model = torch.nn.Linear(3, 2)
    inputs = (torch.randn(4, 3),)
    compiled, parameters, buffers = trace_and_compile(model, inputs, backend="inductor")
    assert received["dynamic"] is True
    assert received["fullgraph"] is True
    if supports_mixed_reduction:
        assert received["options"][key] is False
    else:
        assert key not in received["options"]
    torch.testing.assert_close(
        compiled_call(compiled, model, inputs, parameters, buffers), model(*inputs)
    )


@pytest.mark.parametrize(
    "backend", [None, "eqx", "openequivariance", "cuequivariance", "lunaequivariance"]
)
def test_aoti_dependencies(tmp_path, monkeypatch, backend):
    model = torch.nn.Linear(2, 2)
    expected_libs = set()
    if backend == "eqx":
        import equivariantx as eqx  # noqa: F401

        graph = torch.fx.Graph()
        operator = torch.ops.eqx.gather_sum.default
        inputs = tuple(
            graph.placeholder(arg.name) for arg in operator._schema.arguments
        )
        graph.output(graph.call_function(operator, inputs))
        model = torch.fx.GraphModule(torch.nn.Module(), graph)
        expected_libs = {"equivariantx"}
    elif backend is not None:
        model = type(
            "Convolution",
            (torch.nn.Module,),
            {"__module__": f"tace.models.backends.{backend}._tp_scatter"},
        )()
        expected_libs = {
            "openequivariance": {"openequivariance"},
            "cuequivariance": {"cuequivariance", "cuequivariance_torch"},
            "lunaequivariance": {"tace.models.backends.lunaequivariance"},
        }[backend]
        if backend == "lunaequivariance":
            model.aoti_config = {"dtype": "float64"}
    libs = _custom_ops_libs_from_model(model)
    assert libs == expected_libs

    path = tmp_path / "model.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/archive_format", "pt2")
    _embed_custom_ops_libs(path, libs, model)
    if libs:
        with zipfile.ZipFile(path) as archive:
            assert (
                set(
                    archive.read(f"model/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}")
                    .decode()
                    .split()
                )
                == libs
            )
            if backend == "lunaequivariance":
                assert json.loads(
                    archive.read(f"model/{TACE_AOTI_LUNA_OPS_ENTRY}")
                ) == [model.aoti_config]

    register = Mock()
    if backend == "lunaequivariance":
        from types import SimpleNamespace

        monkeypatch.setitem(
            sys.modules,
            "tace.models.backends.lunaequivariance",
            SimpleNamespace(register_luna_ops=register),
        )
    importer = Mock()
    monkeypatch.setattr("tace.models.compile.aot.importlib.import_module", importer)
    _import_custom_ops_libs(path)
    assert [call.args[0] for call in importer.call_args_list] == sorted(libs)
    if backend == "lunaequivariance":
        register.assert_called_once_with(model.aoti_config)


@pytest.mark.parametrize(
    "backend", ["native", "eqx", "luna", "cartesian", "cartesian_eqx", "cartesian_o2"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_training_and_export(tmp_path, dtype, backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    use_luna = backend == "luna"
    if use_luna:
        if sys.platform != "linux" or torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("LunaEquivariance requires Linux and compute capability >= 8.0")
        pytest.importorskip("LunaEquivariance")
    from tace.models.compile.aot import export_aotinductor, load_aotinductor
    from tace.utils.env import enable_acceleration

    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["atomic_basis"]["type"] = (
        "uv_rra_o2" if backend == "cartesian_o2" else "cgtp"
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    targets = ["energy", "forces", "stress"]
    config["target_property"] = targets
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    enable_acceleration(force=True)
    reference = create_model(config, statistics, targets, []).cuda()
    state = reference.state_dict()
    if backend.startswith("cartesian"):
        from tace.lightning import convert_basis

        state = convert_basis(reference).state_dict()
        config["_target_"] = "tace.models.CartesianTACE"
    enable_acceleration(
        enable_luna=use_luna,
        enable_eqx=backend in ("eqx", "cartesian_eqx", "cartesian_o2"),
        enable_compile=True,
        force=True,
    )
    model = create_model(config, statistics, targets, []).cuda()
    model.load_state_dict(state)
    sample = _synthetic_graph_sample(model)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-9
    for step in range(2):
        data = {key: value.clone() for key, value in sample.items()}
        data["positions"] *= 1 + 0.05 * step
        if step:
            data["edge_index"] = data["edge_index"].flip(1)[:, :3]
            data["edge_shifts"] = data["edge_shifts"].flip(0)[:3]
        expected = reference(dict(data))
        actual = model(dict(data))
        for key in targets:
            assert actual[key] is not None and expected[key] is not None
            torch.testing.assert_close(
                actual[key], expected[key], atol=tolerance, rtol=tolerance
            )
        gradients = [
            torch.autograd.grad(
                sum(out[key].square().sum() for key in targets),
                tuple(module.parameters()),
                allow_unused=True,
            )
            for out, module in ((actual, model), (expected, reference))
        ]
        for got, wanted in zip(*gradients):
            assert (got is None) == (wanted is None)
            if got is not None:
                torch.testing.assert_close(got, wanted, atol=tolerance, rtol=tolerance)
        with torch.no_grad():
            for module, grads in zip((model, reference), gradients):
                for parameter, grad in zip(module.parameters(), grads):
                    if grad is not None:
                        parameter.add_(grad, alpha=-1e-4)
    assert model._compiled_cache

    path = export_aotinductor(model, tmp_path / "model.pt2", sample)
    loaded = load_aotinductor(path, "cuda")
    sample = data  # Use a different edge count from the export example.
    expected = reference.eval()(dict(sample))
    actual = loaded(dict(sample))
    for key in targets:
        torch.testing.assert_close(
            actual[key], expected[key], atol=tolerance, rtol=tolerance
        )
    # Loading must restore registrations without constructing the original model.
    case = tmp_path / "case.pt"
    torch.save(
        {
            "sample": sample,
            "expected": {key: expected[key].detach() for key in targets},
        },
        case,
    )
    script = """
import sys
import torch
from tace.models.compile.aot import load_aotinductor
torch.set_num_threads(4)
case = torch.load(sys.argv[2], weights_only=True)
model = load_aotinductor(sys.argv[1], "cuda")
actual = model(case["sample"])
for key, expected in case["expected"].items():
    torch.testing.assert_close(actual[key], expected,
                               atol=float(sys.argv[3]), rtol=float(sys.argv[3]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(case), str(tolerance)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "TACE_USE_LUNA": "0", "TACE_USE_COMPILE": "0"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_constant_construction_precision():
    from e3nn import o3

    from tace.models.angular import CartesianHarmonics
    from tace.models.blocks import AtomicEnergy, ScaleShift
    from tace.models.linear import e3nnLinear
    from tace.models.radial import (
        AgnesiTransform,
        C2PolynomialCutoff,
        C3PolynomialCutoff,
        CosineCutoff,
        GaussianBasis,
        MollifierCutoff,
        SoftTransform,
        ZBLBasis,
        j0SphericalBesselBasis,
        jnSphericalBesselBasis,
    )
    from tace.models.spherical.basis_change import DirectVirials
    from tace.models.spherical.layer_norm import EquivariantMergeLayerNorm
    from tace.models.spherical.mtace import MagneticBasis
    from tace.models.spherical.symmetric_contraction import Contraction

    constants = []
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        modules = torch.nn.ModuleList(
            [
                j0SphericalBesselBasis(cutoff=4.123456789),
                jnSphericalBesselBasis(cutoff=4.123456789, order=2),
                GaussianBasis(cutoff=4.123456789),
                CosineCutoff(4.123456789),
                MollifierCutoff(4.123456789),
                C2PolynomialCutoff(4.123456789),
                C3PolynomialCutoff(4.123456789),
                AgnesiTransform(),
                SoftTransform(),
                ZBLBasis("cosine"),
                AtomicEnergy([{1: -0.123456789}], [1]),
                ScaleShift([1], [{1: 0.123456789}], [{1: -0.123456789}]),
                MagneticBasis([3.123456789], 4, 1, [1]),
                CartesianHarmonics(3),
                DirectVirials(),
                EquivariantMergeLayerNorm([0, 1, 2], 2),
                e3nnLinear("2x0e + 2x1o", "2x0e + 2x1o"),
                Contraction(
                    o3.Irreps("2x0e + 2x1o"), o3.Irreps("0e"), 2, num_elements=1
                ),
            ]
        )
        buffers = {
            f"{name}.{key}": value
            for name, module in modules.named_modules()
            if type(module).__module__.startswith("tace.")
            for key, value in module.named_buffers(recurse=False)
            if value.is_floating_point()
        }
        assert buffers
        assert all(value.dtype == torch.float64 for value in buffers.values())
        assert all(
            parameter.dtype in (dtype, torch.float64)
            for parameter in modules.parameters()
        )
        constants.append(buffers)

    assert constants[0].keys() == constants[1].keys()
    for name in constants[0]:
        torch.testing.assert_close(
            constants[0][name], constants[1][name], atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_fidelities", [1, 2])
def test_atomic_energy_lookup(dtype, num_fidelities):
    from tace.models.blocks import AtomicEnergy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    elements = list(range(1, 103))
    energies = [
        {z: -100_000.123456789 * z - fidelity for z in elements}
        for fidelity in range(num_fidelities)
    ]
    module = AtomicEnergy(energies, elements).to(device=device)
    node_type = torch.arange(8192, device=device) % len(elements)
    expected = module.atomic_energy.T[node_type]
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for n in (0, 1, 128, 3072, 8192):
            torch.testing.assert_close(
                module(node_type[:n]), expected[:n], atol=0, rtol=0
            )
        compiled = torch.compile(module, fullgraph=True)
        torch.testing.assert_close(compiled(node_type), expected, atol=0, rtol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


@pytest.mark.parametrize("space", ["spherical", "cartesian"])
def test_model_dtype_conversion(space):
    from tace.models.adapter import TensorModel
    from tace.utils.precision import configure_model_precision

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        _target_=f"tace.models.{space.title()}TACE",
        num_layers=1,
        num_channel=2,
        Lmax=1,
        lmax=1,
        cutoff=4.1234567890123,
        max_neighbors=None,
    )
    config["radial_basis"].update(hidden=[4], trainable=True, apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        {
            "atomic_numbers": [1],
            "atomic_energy": {1: -100000.1234567890123},
            "avg_num_neighbors": 2.0,
        }
    ]
    model = create_model(config, statistics, ["energy", "forces"], [])
    readout = model.readout_fn
    atomic_energy = readout.atomic_energy_layer
    atomic_energy.register_buffer("alias", atomic_energy.atomic_energy)
    atomic_energy.float64_buffers = ("atomic_energy", "alias")
    energy = atomic_energy.atomic_energy.clone()
    cutoff = readout.cutoff.clone()
    radial = next(
        module for module in readout.modules() if type(module).__name__ == "RadialBasis"
    )
    physical = next(radial.parameters())
    physical.grad = torch.full_like(physical, 0.1234567890123)
    value, gradient = physical.detach().clone(), physical.grad.clone()
    for wrapper in (TensorModel, CompileTensorModel):
        model = wrapper(readout)
        for dtype, convert in (
            (torch.float32, lambda m: m.float()),
            (torch.float64, lambda m: m.double()),
            (torch.float32, lambda m: m.to(dtype=torch.float32)),
            (torch.float64, lambda m: configure_model_precision(m, torch.float64)),
        ):
            if isinstance(model, CompileTensorModel):
                model._compiled_cache["test"] = None
            assert convert(model) is model
            assert model.get_model_dtype() == dtype
            assert readout.representation.model_dtype == dtype
            assert (
                next(readout.representation.node_embedding.parameters()).dtype == dtype
            )
            assert atomic_energy.atomic_energy is atomic_energy.alias
            torch.testing.assert_close(
                atomic_energy.atomic_energy, energy, atol=0, rtol=0
            )
            torch.testing.assert_close(readout.cutoff, cutoff, atol=0, rtol=0)
            torch.testing.assert_close(physical, value, atol=0, rtol=0)
            torch.testing.assert_close(physical.grad, gradient, atol=0, rtol=0)
            if isinstance(model, CompileTensorModel):
                assert not model._compiled_cache
        with pytest.raises(ValueError, match="use autocast"):
            model.bfloat16()
        assert model.get_model_dtype() == torch.float64
        assert (
            next(readout.representation.node_embedding.parameters()).dtype
            == torch.float64
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "interaction,compiled,skip",
    [
        ("cgtp", False, False),
        ("uv_o2", False, False),
        ("uv_rra_o2", False, False),
        ("cgtp", True, False),
        ("cgtp", True, True),
    ],
)
def test_lightning_model_training_precision(
    monkeypatch, dtype, interaction, compiled, skip
):
    import inspect

    import lightning as L
    from torch.utils.data import DataLoader

    from tace.lightning import create_model
    from tace.lightning.lit_model import LightningWrapperModel

    if compiled:
        if "dynamic_shapes" not in inspect.signature(torch.compile).parameters:
            pytest.skip("Joint training requires explicit dynamic shapes")
        monkeypatch.setenv("TACE_USE_COMPILE", "1")

    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["atomic_basis"]["type"] = interaction
    config["atomic_basis"]["num_head"] = 1
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["apply_cutoff"] = False
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        {"atomic_numbers": [1], "atomic_energy": {1: 0.0}, "avg_num_neighbors": 2.0}
    ]
    model = create_model(config, statistics, ["energy", "forces"], [])
    model.readout_fn.compile_backend = "eager"
    assert all(p.dtype == dtype for p in model.parameters())
    assert model.readout_fn.cutoff.dtype == torch.float64
    assert model.readout_fn.atomic_energy_layer.atomic_energy.dtype == torch.float64
    assert model.readout_fn.atomic_numbers.dtype == torch.int64

    class TrainingModel(LightningWrapperModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            assert loss.dtype == torch.float64 and torch.isfinite(loss)
            if compiled:
                assert len(self.model._compiled_cache) == 1
            return loss

        def on_after_backward(self):
            gradients = [p.grad for p in self.parameters() if p.grad is not None]
            assert gradients
            assert all(g.dtype == dtype and torch.isfinite(g).all() for g in gradients)
            if skip:
                assert all(torch.count_nonzero(g) == 0 for g in gradients)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32-true" if dtype == torch.float32 else "64-true",
        max_steps=2,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    batches = []
    for _ in range(4):
        batch = _magnetic_embedding_sample()
        batch.pop("initial_noncollinear_magmoms")
        batch.update(
            energy=torch.ones(1),
            forces=torch.ones(3, 3),
            entropy=torch.ones(1),
            energy_weight=torch.ones(1),
            forces_weight=torch.ones(1),
        )
        batches.append(batch)
    cfg = dict(
        loss=dict(
            _target_="tace.utils.loss.NormalLoss",
            loss_property=["energy", "forces"],
            loss_function_name=["mse_energy", "mse_forces"],
            loss_property_weights=[1.0, 1.0],
        ),
        optimizer=dict(_target_="torch.optim.SGD", lr=1e-3),
        misc=dict(
            LossSkipController=dict(
                enable=skip,
                manual_threshold=0.0,
                start_step=0,
                ema_window=10,
                multiplier=2.0,
            )
        ),
    )
    module = TrainingModel(cfg, model, ["energy", "forces"], [], statistics)
    trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None))
    assert trainer.global_step == 2


@pytest.mark.parametrize("path_class", [PurePosixPath, PureWindowsPath])
def test_foundation_aoti_output_keeps_version(monkeypatch, path_class):
    from tace.scripts import export_eval

    monkeypatch.setattr(export_eval, "Path", path_class)
    assert (
        export_eval._default_aoti_output_path("TECE-OAM-RRA-1.0")
        == "TECE-OAM-RRA-1.0.pt2"
    )
    assert path_class(
        export_eval._default_aoti_output_path("models/TACE-OAM-7M.pt")
    ) == path_class("models/TACE-OAM-7M.pt2")


@pytest.mark.parametrize("value", [None, "0", "1"])
@pytest.mark.parametrize("training", [False, True])
def test_tf32_environment(monkeypatch, value, training):
    from tace.utils.env import set_tf32
    from tace.utils.utils import set_precision

    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    if value is None:
        monkeypatch.delenv("TACE_USE_TF32", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_TF32", value)
    expected = training if value is None else value == "1"
    configurations = [lambda: set_tf32(training=training)]
    if training:
        configurations.append(lambda: set_precision({"trainer": {"precision": 32}}))
    try:
        for configure in configurations:
            torch.backends.cuda.matmul.allow_tf32 = not expected
            torch.backends.cudnn.allow_tf32 = not expected
            configure()
            assert torch.backends.cuda.matmul.allow_tf32 is expected
            assert torch.backends.cudnn.allow_tf32 is expected
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            previous
        )


@pytest.mark.parametrize(
    "enabled,supported,expected",
    [
        ((), ("eqx", "luna", "oeq", "eqt", "cue"), None),
        (
            ("eqx", "luna", "oeq", "eqt", "cue"),
            ("cue", "eqt", "oeq", "luna", "eqx"),
            "eqx",
        ),
        (("luna", "oeq", "eqt", "cue"), ("cue", "eqt", "oeq", "luna", "eqx"), "luna"),
        (("oeq", "eqt", "cue"), ("cue", "eqt", "oeq", "luna", "eqx"), "oeq"),
        (("eqt", "cue"), ("cue", "eqt", "oeq", "luna", "eqx"), "eqt"),
        (("cue",), ("cue", "eqt", "oeq", "luna", "eqx"), "cue"),
        (("eqx", "luna", "oeq"), ("oeq", "cue"), "oeq"),
        (("eqx", "luna", "oeq"), ("eqt",), None),
    ],
)
def test_acceleration_priority(monkeypatch, enabled, supported, expected):
    from tace.utils.env import ACCELERATION_ENV, select_acceleration

    for name in ("eqx", "luna", "oeq", "eqt", "cue"):
        monkeypatch.setenv(ACCELERATION_ENV[name], "1" if name in enabled else "0")
    monkeypatch.setenv("TACE_USE_COMPILE", "1")
    assert select_acceleration(*supported, kernel="conv") == expected


def test_acceleration_priority_respects_eqx_kernel_switch(monkeypatch):
    from tace.utils.env import EQX_KERNELS, select_acceleration

    monkeypatch.setenv("TACE_USE_EQX", "1")
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    monkeypatch.setitem(EQX_KERNELS, "conv", False)
    assert select_acceleration("eqx", "oeq", kernel="conv") == "oeq"
    assert select_acceleration("eqx", "eqt", kernel="product") == "eqx"


def test_model_input_properties_include_required_embeddings():
    assert {
        "charges",
        "total_charge",
        "initial_collinear_magmoms",
        "initial_noncollinear_magmoms",
    }.issubset(SUPPORTED_EMBEDDING_PROPERTIES)
    cfg = {
        "loss": {
            "loss_property": [
                "charges",
                "collinear_magnetic_forces",
                "noncollinear_magnetic_forces",
            ]
        },
        "model": {
            "config": {
                "universal_embedding": {
                    "electric_field": {"enable": True},
                    "magnetic_field": {"enable": False},
                },
                "atomic_basis": {"type": "cgtp"},
            }
        },
    }

    assert get_embedding_property(cfg) == [
        "electric_field",
        "total_charge",
        "initial_collinear_magmoms",
        "initial_noncollinear_magmoms",
    ]


def test_model_loading_prunes_removed_architecture_keys():
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["atomic_basis"]["removed_atomic_option"] = True
    config["product_basis"]["removed_product_option"] = True
    config["radial_basis"]["unrelated_option"] = True

    cleaned = _prune_removed_keys(config)

    assert "removed_atomic_option" not in cleaned["atomic_basis"]
    assert "removed_product_option" not in cleaned["product_basis"]
    assert cleaned["radial_basis"]["unrelated_option"] is True
    assert "removed_atomic_option" in config["atomic_basis"]
    assert "removed_product_option" in config["product_basis"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interaction_type", ["cgtp", "o2"])
def test_compile_direct_hessian(dtype, interaction_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    targets = ["energy", "forces", "direct_hessian"]
    config["special"]["direct_hessian"]["interaction_type"] = interaction_type
    config.update(
        num_layers=1,
        num_channel=2,
        Lmax=2,
        lmax=2,
        cutoff=4.0,
        max_neighbors=None,
        fidelity=[{"name": "DFT"}],
        target_property=targets,
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    eager = create_model(config, statistics, targets, []).to(device)
    model = CompileTensorModel(deepcopy(eager.readout_fn)).to(device)
    model.readout_fn.compile_backend = "aot_eager"
    sample = _synthetic_graph_sample(model)
    sample["pbc"].zero_()
    expected = eager(dict(sample))
    actual = model(dict(sample))
    assert "direct_hessian_pair_index" not in sample
    assert "direct_hessian_edge_pairs" not in sample
    assert "direct_hessian_edge_shifts" not in sample
    for key in targets:
        torch.testing.assert_close(actual[key], expected[key])
    expected_grads = torch.autograd.grad(
        sum(expected[key].square().sum() for key in targets),
        tuple(eager.parameters()),
        allow_unused=True,
    )
    actual_grads = torch.autograd.grad(
        sum(actual[key].square().sum() for key in targets),
        tuple(model.parameters()),
        allow_unused=True,
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad)

    from tace.utils.loss.normal import NormalLoss

    training_data = model.prepare_property_data(dict(sample))
    training_data.update(
        direct_hessian=torch.zeros_like(actual["direct_hessian"]),
        entropy=torch.ones(2, device=device),
        direct_hessian_weight=torch.ones(2, device=device),
    )
    check_joint_property_loss(
        model,
        training_data,
        NormalLoss(["direct_hessian"], ["mse_direct_hessian"], [1.0]),
    )

    model.eval()
    eager.eval()
    keys = _graph_aoti_input_keys(model)
    outputs = model._output_keys()
    _ensure_sample_inputs(sample, keys, model)
    flat_model = _FlatE3nnCompileModel(model, keys, outputs)
    inputs = tuple(sample[key] for key in keys)
    with _size_oblivious_export():
        traced = trace_to_fx(flat_model, inputs)
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=_graph_dynamic_shapes(keys, num_graphs=2),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    metadata = _export_metadata(model, keys, outputs)
    loaded = AOTICompiledTensorModel(exported.module(), metadata, device)
    sample.pop("direct_hessian_pair_index")
    sample.pop("direct_hessian_edge_pairs")
    sample.pop("direct_hessian_edge_shifts")
    original_positions = sample["positions"].clone()
    for distance in (1.0, model.readout_fn.pair_representation.cutoff + 1):
        sample["positions"] = original_positions * distance
        expected = eager(dict(sample))
        actual = loaded(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])
        assert "direct_hessian_pair_index" not in sample
        assert "direct_hessian_edge_pairs" not in sample
        assert actual["direct_hessian"].shape == (8, 3, 3)
        if distance > 1:
            assert not actual["direct_hessian"][[1, 2, 5, 6]].any()

    isolated = {key: value[:2] for key, value in sample.items()}
    isolated["ptr"] = sample["ptr"].new_tensor([0, 1, 2])
    isolated["batch"] = sample["batch"].new_tensor([0, 1])
    isolated["edge_index"] = sample["edge_index"][:, :0]
    isolated["edge_shifts"] = sample["edge_shifts"][:0]
    expected = eager(dict(isolated))
    actual = loaded(dict(isolated))
    for key in targets:
        torch.testing.assert_close(actual[key], expected[key])
    assert actual["direct_hessian"].shape == (2, 3, 3)
    isolated = {key: value[:1] for key, value in isolated.items()}
    isolated["ptr"] = sample["ptr"].new_tensor([0, 1])
    isolated["edge_index"] = sample["edge_index"][:, :0]
    with pytest.raises(ValueError, match="at least two atoms"):
        loaded(isolated)

    sample["positions"] = original_positions
    sample["lattice"][0] = sample["lattice"].new_tensor(
        [[12, 0, 0], [2, 14, 0], [1, 2, 16]]
    )
    sample["pbc"][0, 0] = True
    expected = eager(dict(sample))
    for module in (model, loaded):
        actual = module(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])
    # Moving one atom by a lattice vector must leave all predictions unchanged.
    sample["positions"] = original_positions.clone()
    sample["positions"][1] += sample["lattice"][0, 0]
    sample["edge_shifts"][:2, 0] = sample["edge_shifts"].new_tensor([-1, 1])
    for module in (eager, model, loaded):
        actual = module(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])

    model.reset_target_property(["direct_hessian"])
    assert model._output_keys() == ("direct_hessian",)


class _MagneticEmbeddingReadout(torch.nn.Module):
    def __init__(
        self,
        embedding_property: list[str],
        atomic_basis_type: Union[str, None],
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer("cutoff", torch.tensor(4.0))
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.int64))
        self.target_property = ["energy", "forces"]
        self.embedding_property = embedding_property
        self.model_config = (
            {"atomic_basis": {"type": atomic_basis_type}}
            if atomic_basis_type is not None
            else {}
        )
        self.max_neighbors = None

    def forward(self, data, graph):
        node_energy = self.scale * (
            graph.positions.square().sum(dim=-1)
            + data["initial_noncollinear_magmoms"].square().sum(dim=-1)
        )
        energy = torch.zeros(
            graph.num_graphs,
            dtype=node_energy.dtype,
            device=node_energy.device,
        ).index_add(0, data["batch"], node_energy)
        return {"energy": energy, "node_energy": node_energy}


class _ChargeReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer("cutoff", torch.tensor(4.0))
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.int64))
        self.target_property = ["charges"]
        self.embedding_property = []
        self.model_config = {}
        self.max_neighbors = None

    def forward(self, data, graph):
        raw_charges = self.scale * graph.positions[:, 0]
        graph_charges = torch.zeros_like(data["total_charge"]).index_add(
            0,
            data["batch"],
            raw_charges,
        )
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]
        correction = (data["total_charge"] - graph_charges) / num_atoms
        return {
            "energy": graph_charges,
            "charges": raw_charges + correction[data["batch"]],
        }


def _magnetic_embedding_sample() -> dict[str, torch.Tensor]:
    return {
        "pbc": torch.ones(1, 3, dtype=torch.bool),
        "positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "node_type": torch.zeros(3, dtype=torch.int64),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_shifts": torch.zeros(4, 3),
        "lattice": torch.eye(3).unsqueeze(0) * 10.0,
        "batch": torch.zeros(3, dtype=torch.int64),
        "ptr": torch.tensor([0, 3], dtype=torch.int64),
        "fidelity_idx": torch.zeros(1, dtype=torch.int64),
        "initial_noncollinear_magmoms": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        ),
    }


def test_aoti_charges_requires_total_charge_input():
    import inspect

    model = CompileTensorModel(_ChargeReadout()).eval()
    sample = _magnetic_embedding_sample()
    sample.pop("initial_noncollinear_magmoms")

    with pytest.raises(KeyError, match="total_charge"):
        model._input_keys(sample)

    input_keys = _graph_aoti_input_keys(model)
    assert input_keys[-1] == "total_charge"
    with pytest.raises(KeyError, match="total_charge"):
        _ensure_sample_inputs(dict(sample), input_keys, model)

    sample["total_charge"] = torch.tensor([1.5])
    assert model._input_keys(sample)[-1] == "total_charge"
    output_keys = model._output_keys()
    assert output_keys == ("charges",)
    flat_model = _FlatE3nnCompileModel(model, input_keys, output_keys).eval()
    inputs = tuple(sample[key] for key in input_keys)
    (charges,) = flat_model(*inputs)
    torch.testing.assert_close(charges.sum(), sample["total_charge"].sum())
    traced_model = trace_to_fx(flat_model, inputs)
    (traced_charges,) = traced_model(*inputs)
    torch.testing.assert_close(traced_charges, charges)

    synthetic = _synthetic_graph_sample(model)
    assert synthetic["total_charge"].shape == (2,)
    metadata = _export_metadata(model, input_keys, output_keys)
    assert "total_charge" in json.loads(metadata["tace_input_keys"])
    assert "total_charge" in json.loads(metadata["tace_embedding_property"])
    assert json.loads(metadata["tace_output_keys"]) == ["charges"]

    from tace.utils.loss.normal import NormalLoss

    model.readout_fn.scale.requires_grad_(True)
    loss_fn = NormalLoss(["charges"], ["mse_charges"], [1.0])
    for n in (3, 5, 1):
        sample.update(
            positions=torch.randn(n, 3),
            node_type=torch.zeros(n, dtype=torch.long),
            batch=torch.zeros(n, dtype=torch.long),
            ptr=torch.tensor([0, n]),
            charges=torch.zeros(n),
            entropy=torch.ones(1),
            charges_weight=torch.ones(1),
        )
        if n == 1:
            sample.update(
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_shifts=torch.empty(0, 3),
            )
        check_joint_property_loss(model, sample, loss_fn)
        if "dynamic_shapes" in inspect.signature(torch.compile).parameters:
            assert len(model._compiled_cache) == (2 if n == 1 else 1)


def check_joint_property_loss(model, data, loss_fn):
    import inspect

    from tace.models.adapter import TensorModel

    if "dynamic_shapes" not in inspect.signature(torch.compile).parameters:
        return
    training = model.training
    backend = getattr(model.readout_fn, "compile_backend", "inductor")
    model.train()
    model.readout_fn.compile_backend = "eager"
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    expected = TensorModel.forward(model, dict(data))
    expected_loss = loss_fn(expected, data)
    expected_grads = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    _, loss = model.forward_loss(dict(data), loss_fn)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    torch.testing.assert_close(loss, expected_loss)
    for actual, expected in zip(gradients, expected_grads):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
    model.train(training)
    model.readout_fn.compile_backend = backend


def test_compiled_evaluation_disables_autograd(monkeypatch):
    model = CompileTensorModel(_ChargeReadout()).eval()
    sample = _magnetic_embedding_sample()
    sample.pop("initial_noncollinear_magmoms")
    sample["total_charge"] = torch.tensor([1.5])
    observed = []

    monkeypatch.setattr(
        compile_wrapper,
        "trace_and_compile",
        lambda *args, **kwargs: (object(), (), ()),
    )

    def fake_compiled_call(*args, **kwargs):
        observed.append(torch.is_grad_enabled())
        return (torch.zeros(3),)

    monkeypatch.setattr(compile_wrapper, "compiled_call", fake_compiled_call)

    with torch.enable_grad():
        model._compiled_forward(sample)

    assert observed == [False]


def _slice_update(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output = x.clone()
    output[:, 1:2, :] = value
    return output


def test_trace_to_fx_removes_full_dynamic_slice_scatter():
    x = torch.randn(4, 3, 5)
    value = torch.randn(4, 1, 5)
    traced = trace_to_fx(_slice_update, (x, value), functionalize=True)

    for node in traced.graph.nodes:
        if node.op != "call_function":
            continue
        assert node.target != torch.ops.aten.copy_.default
        if node.target == torch.ops.aten.slice_scatter.default:
            start = node.args[3] if len(node.args) > 3 else None
            end = node.args[4] if len(node.args) > 4 else None
            assert (start, end) != (0, sys.maxsize)

    torch.testing.assert_close(traced(x, value), _slice_update(x, value))


@pytest.mark.parametrize(
    ("embedding_property", "atomic_basis_type"),
    [
        pytest.param([], "uv_o2_mag", id="o2-magnetic"),
        pytest.param(
            ["initial_noncollinear_magmoms"],
            "cgtp",
            id="universal-embedding",
        ),
    ],
)
def test_aoti_keeps_magnetic_embedding_without_magnetic_force_target(
    embedding_property,
    atomic_basis_type,
):
    model = CompileTensorModel(
        _MagneticEmbeddingReadout(embedding_property, atomic_basis_type)
    ).eval()
    sample = _magnetic_embedding_sample()

    assert "initial_noncollinear_magmoms" in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )
    assert "noncollinear_magnetic_forces" not in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )

    input_keys = _graph_aoti_input_keys(model)
    output_keys = model._output_keys()
    assert "initial_noncollinear_magmoms" in input_keys
    assert "noncollinear_magnetic_forces" not in output_keys

    flat_model = _FlatE3nnCompileModel(model, input_keys, output_keys).eval()
    output = flat_model(*(sample[key] for key in input_keys))
    zero_magmoms = dict(sample)
    zero_magmoms["initial_noncollinear_magmoms"] = torch.zeros_like(
        sample["initial_noncollinear_magmoms"]
    )
    output_without_magmoms = flat_model(*(zero_magmoms[key] for key in input_keys))
    assert output_keys == ("energy", "node_energy", "forces")
    assert not torch.equal(output[0], output_without_magmoms[0])
    torch.testing.assert_close(output[2], output_without_magmoms[2])
    torch.testing.assert_close(output[2], -2.0 * sample["positions"])

    metadata = _export_metadata(model, input_keys, output_keys)
    assert "initial_noncollinear_magmoms" in json.loads(metadata["tace_input_keys"])
    assert "initial_noncollinear_magmoms" in json.loads(
        metadata["tace_embedding_property"]
    )
    assert "noncollinear_magnetic_forces" not in json.loads(
        metadata["tace_output_keys"]
    )

    from tace.utils.loss.normal import NormalLoss

    model.readout_fn.scale.requires_grad_(True)
    model.reset_target_property(["energy", "noncollinear_magnetic_forces"])
    sample.update(
        noncollinear_magnetic_forces=torch.zeros(3, 3),
        entropy=torch.ones(1),
        noncollinear_magnetic_forces_weight=torch.ones(1),
    )
    check_joint_property_loss(
        model,
        sample,
        NormalLoss(
            ["noncollinear_magnetic_forces"],
            ["mse_noncollinear_magnetic_forces"],
            [1.0],
        ),
    )


def test_import_without_cpu_affinity_or_cuda(tmp_path):
    script = """
import os
import sys

if hasattr(os, "sched_getaffinity"):
    del os.sched_getaffinity
sys.modules["torch.utils.cpp_extension"] = None

from tace.interface.ase import TACEAseCalc
from tace.lightning import load_tace
from tace.scripts.train import main
from equivariantx.backends.cuda import runtime as cuda

assert not hasattr(os, "sched_getaffinity")
assert not cuda._POOL._threads
assert not cuda.runtime.cache_info().currsize
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("training", [False, True])
def test_precision_policy(monkeypatch, training):
    from tace.utils.precision import PrecisionPolicy, resolve_precision

    monkeypatch.delenv("TACE_USE_TF32", raising=False)
    monkeypatch.delenv("TACE_MATMUL_PRECISION", raising=False)
    default_dtype = torch.get_default_dtype()
    cfg = {"dtype": {"model": "float32", "autocast": None, "matmul": None}}
    policy = resolve_precision(cfg, training=training)
    assert policy.matmul == ("tf32" if training else "ieee")
    assert policy.torch_dtype == torch.float32
    assert torch.get_default_dtype() == default_dtype
    assert cfg["dtype"]["matmul"] == policy.matmul

    mixed = resolve_precision({"dtype": {"autocast": "bfloat16"}}, training=training)
    assert mixed.matmul == "ieee"
    for options in (
        {"model": "bfloat16"},
        {"model": "float64", "autocast": "bfloat16"},
        {"autocast": "float16", "matmul": "tf32"},
        {"matmul": "bf16x3"},
    ):
        with pytest.raises(ValueError):
            PrecisionPolicy(**options)

    monkeypatch.setenv("TACE_USE_TF32", "0")
    assert resolve_precision({}, training=training).matmul == "ieee"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_precision({"dtype": {"matmul": "tf32"}}, training=training)
    monkeypatch.setenv("TACE_MATMUL_PRECISION", "tf32")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_precision({}, training=training)


def test_model_dtype_metadata(tmp_path, monkeypatch):
    from tace.lightning import export_tace, load_tace

    monkeypatch.setenv("TACE_USE_TF32", "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=1,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [4]
    model = create_model(
        model_config=config,
        statistics=[
            {
                "atomic_numbers": [1],
                "atomic_energy": {1: -1.23456789123},
                "avg_num_neighbors": 2.0,
                "rms_forces": {1: 1.0},
            }
        ],
        target_property=["energy"],
        embedding_property=[],
    )
    path = tmp_path / "model.pt"
    export_tace(model, str(path))
    package = torch.load(path, weights_only=False)
    assert package["dtype"]["model"] == str(model.get_model_dtype()).removeprefix(
        "torch."
    )
    restored = load_tace(path)
    assert restored.get_model_dtype() == model.get_model_dtype()
    assert restored.readout_fn.atomic_energy_layer.atomic_energy.dtype == torch.float64
    with pytest.raises(ValueError, match="FP64 geometry"):
        load_tace(path, device="mps")
