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
        valid = (
            torch.arange(data["positions"].shape[0], device=data["positions"].device)
            < data["ptr"][-1]
        )
        node_energy = torch.where(valid, result["node_energy"], 0)
        result["node_energy"] = node_energy
        result["energy"] = node_energy.new_zeros((data["ptr"].numel() - 1,)).index_add(
            0, data["batch"], node_energy
        )
        return result


def enable_padded_readout(readout):
    """Change only the copied readout's execution; retain all parameters."""
    if set(readout.target_property) != {"energy", "forces", "stress"}:
        raise ValueError("Padded OAM execution supports only energy, forces and stress")
    if type(readout) is not SphericalTACE:
        raise TypeError("Padded execution requires the validated SphericalTACE readout")
    readout.__class__ = PaddedSphericalTACE
    readout.oam_xl_padded_batches = True


def pad_batch(batch, node_multiple, edge_multiple, *, cutoff=6.0):
    """Round a batch's capacities without changing its physical graphs.

    At least 32 dummy atoms distribute the dummy edges and prevent a single
    exceptionally long CSR row. Dummy edges join disjoint atom pairs beyond
    ``cutoff``, independent of the physical cells.
    """
    if node_multiple < 1 or edge_multiple < 1:
        raise ValueError("Padding multiples must be positive")
    nodes, edges = batch["positions"].shape[0], batch["edge_index"].shape[1]
    padded_nodes = ((nodes + 32 + node_multiple - 1) // node_multiple) * node_multiple
    padded_edges = ((edges + edge_multiple - 1) // edge_multiple) * edge_multiple
    extra_nodes, extra_edges = padded_nodes - nodes, padded_edges - edges
    result = dict(batch)
    for key in (
        "edge_pair_rows",
        "edge_pair_inverse",
        "oam_radial_representatives",
        "oam_radial_map",
    ):
        result.pop(key, None)
    node_keys = {
        "atomic_numbers",
        "positions",
        "node_type",
        "num_neighbors",
        "forces",
        "batch",
    }
    for key in node_keys & batch.keys():
        value = batch[key]
        if value.shape[0] != nodes:
            raise ValueError(f"Unexpected node field shape: {key}")
        extra = value.new_zeros((extra_nodes, *value.shape[1:]))
        if key in {"atomic_numbers", "node_type"}:
            extra[:] = value[0]
        elif key == "batch":
            extra[:] = batch["ptr"].numel() - 2
        elif key == "positions" and extra_edges:
            extra[1::2, 0] = cutoff + 1.0
        result[key] = torch.cat((value, extra))
    # Keep complete pairs even when the number of padding atoms is odd.
    dummy = torch.arange(extra_edges, device=batch["edge_index"].device) % (
        extra_nodes // 2 * 2
    )
    result["edge_index"] = torch.cat(
        (batch["edge_index"], torch.stack((dummy, dummy ^ 1)) + nodes), dim=1
    )
    shifts = batch["edge_shifts"].new_zeros((extra_edges, 3))
    result["edge_shifts"] = torch.cat((batch["edge_shifts"], shifts))
    result["num_nodes"] = padded_nodes
    return result


def padded_force_loss(loss):
    """Exclude trailing padding from the per-atom L2 force loss."""
    if "l2mae_forces" not in loss.loss_function_name:
        raise ValueError("Padding requires the original L2MAE force objective")
    for name, kwargs in zip(
        loss.loss_function_name, loss.loss_function_kwargs, strict=True
    ):
        if name == "l2mae_forces":
            kwargs["padded"] = True
    return loss
