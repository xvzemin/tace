################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from copy import deepcopy

import torch


def convert_basis(model, basis="cartesian"):
    """Copy a model in an equivalent tensor representation.

    Parameters
    ----------
    model : tace.models.TensorModel
        Eager model with matching coupling paths and architecture.
    basis : {"cartesian", "spherical"}, optional
        Storage for the returned model. Parameters are copied without
        reinitialization; representation-dependent constants are rebuilt.

    Returns
    -------
    tace.models.TensorModel
        Independent model preserving dtype, device, heads and training mode.
        Recreate the optimizer before further training.
    """
    from equivariantx.o3 import TransposeIrrepsLayout
    from tace.lightning.torch_model import create_model
    from tace.models.adapter import TensorModel
    from tace.models.linear import switch_e3nn_weight_layout
    from tace.utils.precision import configure_model_precision, torch_default_dtype

    if not isinstance(model, TensorModel):
        raise TypeError("Pass an eager TensorModel, for example from load_tace.")
    if basis not in ("cartesian", "spherical"):
        raise ValueError("basis must be 'cartesian' or 'spherical'.")
    config = deepcopy(model.readout_fn.model_config)
    config["_target_"] = (
        "tace.models.CartesianTACE"
        if basis == "cartesian"
        else "tace.models.SphericalTACE"
    )
    if basis == "spherical":
        kinds = config["product_basis"]["type"]
        config["product_basis"]["type"] = (
            ["nonlinear_cgtp" if kind == "nonlinear_ictd" else kind for kind in kinds]
            if isinstance(kinds, list)
            else ("nonlinear_cgtp" if kinds == "nonlinear_ictd" else kinds)
        )
    with (
        torch.random.fork_rng(devices=[]),
        torch_default_dtype(model.get_model_dtype()),
    ):
        result = create_model(
            config,
            model.readout_fn.statistics,
            model.get_target_property(),
            model.get_embedding_property(),
        )
    configure_model_precision(
        result, model.get_model_dtype(), device=model.readout_fn.cutoff.device
    )
    source_modules = dict(model.named_modules())
    for name, module in result.named_modules():
        original = source_modules.get(name)
        if hasattr(module, "use_matrix_weight") and hasattr(
            original, "use_matrix_weight"
        ):
            switch_e3nn_weight_layout(
                module, "matrix" if original.use_matrix_weight else "flat"
            )
    source, target = dict(model.named_parameters()), dict(result.named_parameters())
    if source.keys() != target.keys() or any(
        source[k].shape != v.shape for k, v in target.items()
    ):
        raise ValueError(
            "Basis conversion requires identical parameter names and shapes."
        )
    constants = set()
    for representation in (model, result):
        for name, module in representation.named_modules():
            if module.__module__.startswith(
                (
                    "equivariantx.co3",
                    "equivariantx.co2",
                    "tace.models.cartesian.local",
                    "tace.models.cartesian.layers",
                    "tace.models.cartesian.prod",
                )
            ) or isinstance(module, TransposeIrrepsLayout):
                constants.update(
                    f"{name}.{key}" for key, _ in module.named_buffers(recurse=False)
                )
    state, previous = result.state_dict(), model.state_dict()
    for name, value in previous.items():
        if name.endswith(".block_weights") or (
            name in state and name not in constants and state[name].shape == value.shape
        ):
            state[name] = value
    result.load_state_dict(state, strict=True)
    for name, parameter in target.items():
        parameter.requires_grad_(source[name].requires_grad)
    result.train(model.training)
    result.reset_fidelity_idx(model.get_fidelity_idx())
    result._set_lammps_mliap(model.lmp)
    result.retain_graph, result.create_graph = model.retain_graph, model.create_graph
    return result
