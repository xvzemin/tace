"""Single-rank ML-IAP execution of the periodic OAM-L force graph."""

from pathlib import Path

import torch

from equivariantx_foundation.targets.lammps.local_owner import LocalOwnerForceRunner
from tace.interface.lammps.mliap import TACEAOTILammpsCalc


def export_lammps(model, output, *, capacity, aoti_package=None):
    """Export a local-owner force package and optional receiver capacity model."""
    from .tangent import (
        _frozen_oam_l_copy,
        _tangent_from_frozen,
        export_tangent_lammps,
    )

    if not capacity:
        return export_tangent_lammps(model, output, aoti_package=aoti_package)

    from tace.models.compile import export_lammps_aotinductor

    from .model import optimize_oam_l
    from .receiver import CapacityOAMLammpsCalc

    output = Path(output)
    streamed = (
        Path(aoti_package) if aoti_package is not None else output.with_suffix(".pt2")
    )
    if streamed.suffix != ".pt2":
        streamed = Path(str(streamed) + ".pt2")
    compact = streamed.with_name(streamed.stem + "-compact.pt2")
    if output.resolve() in (streamed.resolve(), compact.resolve()):
        raise ValueError(
            "Use distinct paths for the ML-IAP loader and its .pt2 packages."
        )
    model = _frozen_oam_l_copy(model)
    lowered = _tangent_from_frozen(model, keep_base=True)
    export_lammps_aotinductor(lowered, streamed, local_owners=True)
    del lowered
    lowered = optimize_oam_l(
        model,
        memory_mode="recompute",
        fused_geometry=True,
        fused_projection=False,
        grouped_coefficients=False,
    )
    export_lammps_aotinductor(lowered, compact, local_owners=True)
    del lowered
    loader = CapacityOAMLammpsCalc.from_packages(
        streamed,
        compact,
        model.get_atomic_numbers(),
        model.get_cutoff(),
        model.get_model_dtype(),
    ).attach_frozen_receiver(
        optimize_oam_l(model, fused_geometry=True, fused_projection=True)
    )
    loader.receiver_tangent = True
    loader.domain_receiver_tangent = True
    torch.save(loader, output, pickle_protocol=5)
    return output


class OAMLammpsCalc(LocalOwnerForceRunner, TACEAOTILammpsCalc):
    """Evaluate a local-owner force package on a single LAMMPS rank."""

    graph_max_atoms = 0
    graph_max_edges = 65_536
    compact_package = None

    @classmethod
    def from_packages(cls, streamed, compact, atomic_numbers, cutoff, dtype):
        """Package two frozen force graphs and select one at LAMMPS initialization."""
        calc = cls(streamed, atomic_numbers, cutoff, dtype)
        compact = Path(compact)
        calc.compact_package = (compact.name, compact.read_bytes())
        calc.graph_max_atoms = 1024
        return calc

    def _load_local_package(self, data):
        if self.compact_package is not None:
            device = torch.as_tensor(data.elems).device
            free, _ = torch.cuda.mem_get_info(device)
            # Budget includes five live radial states, node states, force VJP and
            # allocator headroom. Only the selected package is loaded on the GPU.
            if data.npairs * 12288 + (2 << 30) > free:
                self.package_filename, self.package_bytes = self.compact_package
            print(f"EQX OAM-L force package: {self.package_filename}", flush=True)

        super()._load_local_package(data)

    def compute_forces(self, data):
        n, e = data.nlocal, data.npairs
        if n == 0 or (n == 1 and e == 0):
            # For exactly one edgeless atom the released checkpoint disables
            # the learned scale/shift. Adding a disconnected atom would change
            # that physical energy. The legacy PT2 ABI has no isolated-energy
            # table and cannot evaluate the unpadded shape.
            raise ValueError(
                "OAM-L ML-IAP requires a nonempty system other than a single "
                "isolated atom; use eager TACE for its isolated-atom energy."
            )
        super().compute_forces(data)


class RadialTangentLammpsCalc(OAMLammpsCalc):
    """Single-rank loader for a frozen radial-tangent OAM-L force package."""

    def _initialize_device(self, data):
        from . import tangent  # noqa: F401

        if data.nlocal > self.graph_max_atoms or data.npairs > self.graph_max_edges:
            # Large systems keep the standard AOTI runner. Capture requires
            # the single-runner loader selected during initialization.
            self.graph_max_atoms = 0
        super()._initialize_device(data)
