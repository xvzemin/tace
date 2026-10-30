"""Conversion between spherical and Cartesian TACE models."""

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
    from tace.lightning.torch_model import create_model
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
    from tace.models.adapter import TensorModel
    from tace.models.layout import LayoutTransform
    from tace.models.linear import switch_e3nn_weight_layout
    from tace.utils.utils import torch_default_dtype

    if not isinstance(model, TensorModel):
        raise TypeError("Pass an eager TensorModel, for example from load_tace.")
    if basis not in ("cartesian", "spherical"):
        raise ValueError("basis must be 'cartesian' or 'spherical'.")
    config = deepcopy(model.readout_fn.model_config)
    config["_target_"] = (
        "tace.models.cartTACE" if basis == "cartesian" else "tace.models.e3nnTACE"
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
    result.to(device=model.readout_fn.cutoff.device, dtype=model.get_model_dtype())
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
                    "eqx.co3",
                    "eqx.co2",
                    "tace.models._cart.local",
                    "tace.models._cart.layers",
                )
            ) or isinstance(module, (LayoutTransform, EquivariantMergeLayerNorm)):
                constants.update(
                    f"{name}.{key}" for key, _ in module.named_buffers(recurse=False)
                )
    state, previous = result.state_dict(), model.state_dict()
    for name in state.keys() & previous.keys():
        if name not in constants and state[name].shape == previous[name].shape:
            state[name] = previous[name]
    for name, module in result.named_modules():
        if not isinstance(module, EquivariantMergeLayerNorm):
            continue
        original = source_modules[name]
        if (
            module.balance_degree_weight is None
            or original.balance_degree_weight is None
        ):
            continue
        source_widths = [
            3**l if original.dim == sum(3**k for k in original.ls) else 2 * l + 1
            for l in original.ls
        ]
        target_widths = [
            3**l if module.dim == sum(3**k for k in module.ls) else 2 * l + 1
            for l in module.ls
        ]
        values = original.balance_degree_weight.split(source_widths, dim=-1)
        state[name + ".balance_degree_weight"] = torch.cat(
            [
                value[..., :1].expand(*value.shape[:-1], width)
                for value, width in zip(values, target_widths)
            ],
            -1,
        )
    result.load_state_dict(state, strict=True)
    for name, parameter in target.items():
        parameter.requires_grad_(source[name].requires_grad)
    result.train(model.training)
    result.reset_fidelity_idx(model.get_fidelity_idx())
    result._set_lammps_mliap(model.lmp)
    result.retain_graph, result.create_graph = model.retain_graph, model.create_graph
    return result
