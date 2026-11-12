"""Bounded graph shapes with exactly masked dummy atoms for E/F/S training.

The real ``ptr`` and graph labels are retained. Dummy atoms have no connection
to real atoms, contribute no energy, and are excluded from force-loss counts.
This module is specific to the validated OAM energy/force/stress contract.
"""

import torch
from tace.models.spherical.tace import SphericalTACE


class PaddedSphericalTACE(SphericalTACE):
    """Mask dummy energies before differentiating for forces and stress."""

    def readout_fn(self, data, graph, representation, edge_features):
        result = super().readout_fn(data, graph, representation, edge_features)
        valid = torch.arange(data["positions"].shape[0], device=data["positions"].device) < data["ptr"][-1]
        node_energy = torch.where(valid, result["node_energy"], 0)
        result["node_energy"] = node_energy
        result["energy"] = node_energy.new_zeros((data["ptr"].numel() - 1,)).index_add(0, data["batch"], node_energy)
        return result


def enable_padded_readout(readout):
    """Change only the copied readout's execution; retain all parameters."""
    if set(readout.target_property) != {"energy", "forces", "stress"}:
        raise ValueError("Padded OAM execution supports only energy, forces and stress")
    if type(readout) is not SphericalTACE:
        raise TypeError("Padded execution requires the validated SphericalTACE readout")
    readout.__class__ = PaddedSphericalTACE
    readout.oam_xl_padded_batches = True


def pad_batch(batch, node_multiple, edge_multiple):
    """Round a batch's capacities without changing its physical graphs.

    At least 32 dummy atoms distribute the dummy edges and prevent a single
    exceptionally long CSR row. Periodic self edges have a nonzero vector,
    avoiding singular spherical harmonics even though their energies are masked.
    """
    if node_multiple < 1 or edge_multiple < 1:
        raise ValueError("Padding multiples must be positive")
    nodes, edges = batch["positions"].shape[0], batch["edge_index"].shape[1]
    padded_nodes = ((nodes + 32 + node_multiple - 1) // node_multiple) * node_multiple
    padded_edges = ((edges + edge_multiple - 1) // edge_multiple) * edge_multiple
    extra_nodes, extra_edges = padded_nodes - nodes, padded_edges - edges
    result = dict(batch)
    node_keys = {"atomic_numbers", "positions", "node_type", "num_neighbors", "forces", "batch"}
    for key in node_keys & batch.keys():
        value = batch[key]
        if value.shape[0] != nodes:
            raise ValueError(f"Unexpected node field shape: {key}")
        extra = value.new_zeros((extra_nodes, *value.shape[1:]))
        if key in {"atomic_numbers", "node_type"}:
            extra[:] = value[0]
        elif key == "batch":
            extra[:] = batch["ptr"].numel() - 2
        result[key] = torch.cat((value, extra))
    dummy = torch.arange(extra_edges, device=batch["edge_index"].device) % extra_nodes + nodes
    result["edge_index"] = torch.cat((batch["edge_index"], torch.stack((dummy, dummy))), dim=1)
    shifts = batch["edge_shifts"].new_zeros((extra_edges, 3))
    shifts[:, 0] = 1
    result["edge_shifts"] = torch.cat((batch["edge_shifts"], shifts))
    result["num_nodes"] = padded_nodes
    return result


def register_padded_force_loss(loss):
    """Retain the original weighted per-real-atom L2 force loss and its class."""
    from tace.utils.loss.common import apply_element_weights
    from tace.utils.loss.registry import register_loss

    @register_loss(key="oam_xl_padded_l2mae_forces")
    def padded_force_loss(pred, label, element_weights=None):
        valid = torch.arange(pred["forces"].shape[0], device=pred["forces"].device) < label["ptr"][-1]
        error = torch.where(valid[:, None], pred["forces"] - label["forces"], 1)
        weight = (label["entropy"] * label["forces_weight"])[label["batch"]]
        weight = apply_element_weights(weight, label, element_weights)
        return (torch.linalg.vector_norm(error, dim=-1) * weight * valid).sum() / label["ptr"][-1]

    names = list(loss.loss_function_name)
    if "l2mae_forces" not in names:
        raise ValueError("Padding requires the original L2MAE force objective")
    loss.loss_function_name = ["oam_xl_padded_l2mae_forces" if name == "l2mae_forces" else name for name in names]
    return loss
