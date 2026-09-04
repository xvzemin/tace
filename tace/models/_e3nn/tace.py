################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Any, Dict, List, Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from ..blocks import OneHotToAtomicEnergy, ScaleShift
from ..linear import e3nnLinear
from ..radial import ZBLBasis
from ..utils import compute_fixed_charge_dipole, get_target_irreps
from .basis_change import DirectPolarizability, DirectVirials
from .default import check_model_config
from .les import TACELES, required_les_irreps
from .readout import (
    build_element_scalar_readout,
    build_scalar_readout,
    build_tensor_readout,
)
from .representation import Representation


class e3nnTACE(torch.nn.Module):
    def __init__(
        self,
        Lmax: int,
        lmax: int,
        num_layers: int,
        num_channel: int,
        cutoff: float,
        statistics: List[Dict],
        target_property: List[str],
        max_neighbors: Union[int, None],
        fidelity: Dict,
        node_embedding: Dict,
        edge_embedding: Dict,
        edge_update: Dict,
        radial_basis: Dict,
        atomic_basis: Dict,
        product_basis: Dict,
        readout_emlp: Dict,
        scale_shift: Dict,
        short_range: Dict,
        long_range: Dict,
        universal_embedding: Dict,
        magnetic_edge_update: Dict = {},
        special: Dict = {},
        resnet: Dict = {},
        layer_norm: Dict = {},
        normalizer: Dict = {},
        parity: bool = False,
        mmax: int = 2,
        mag_Lmax: int = 1,
        dropout: Dict = {},
        embedding_property=[],
        atomic_numbers=None,
        **kwargs,
    ):
        cfg = {
            k: v
            for k, v in locals().items()
            if k == "_target_" or (k != "self" and not k.startswith("_"))
        }
        # self.model_config = locals()
        # del self.config['self']
        self.model_config = cfg
        cfg = check_model_config(cfg)
        super().__init__()

        # === Will be called by other module ===
        self.fidelity = cfg["fidelity"]
        self.num_fidelities = len(cfg["fidelity"])
        self.max_neighbors = cfg["max_neighbors"]
        self.target_property = cfg["target_property"]
        self.num_layers = cfg["num_layers"]
        self.statistics = cfg["statistics"]
        self.special = cfg["special"]
        self.embedding_property = (
            cfg["invariant_property"] + cfg["equivariant_property"]
        )
        magnetic_interactions = {"o2_mag"}
        self.use_one_body_magmoms = bool(
            cfg["readout_emlp"]["use_one_body_magmoms"]
            and "energy" in cfg["target_property"]
            and (
                "initial_noncollinear_magmoms" in cfg["embedding_property"]
                or "initial_noncollinear_magmoms" in self.embedding_property
                or any(
                    interaction in magnetic_interactions
                    for interaction in cfg["atomic_basis"]["type"]
                )
            )
        )
        if self.use_one_body_magmoms:
            self.embedding_property = list(
                dict.fromkeys(
                    self.embedding_property + ["initial_noncollinear_magmoms"]
                )
            )
        self.register_buffer(
            "cutoff", torch.tensor(cfg["cutoff"], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "atomic_numbers", torch.tensor(cfg["atomic_numbers"], dtype=torch.int64)
        )

        # === Will be called by in this module ===
        self.use_alllayer = cfg["readout_emlp"]["use_alllayer"]
        self.num_elements = len(cfg["atomic_numbers"])
        self.num_channel = cfg["num_channel"]
        self.scale_zbl = cfg["scale_shift"]["scale_zbl"]

        # === Will be used in __init__ ===
        target_irreps = get_target_irreps(self.target_property)
        les_cfg = cfg["long_range"]["les"]
        if les_cfg["enable"]:
            if "energy" not in self.target_property:
                raise ValueError("LES requires energy in target_property")
            les_arguments = les_cfg["les_arguments"] or {}
            target_irreps.extend(required_les_irreps(les_arguments))
        if cfg["product_basis"]["return_components"] is not None:
            target_irreps.extend(cfg["product_basis"]["return_components"])
        target_irreps = list(set(target_irreps))
        self.target_irreps = o3.Irreps(target_irreps).regroup()

        # === Representation/Descriptor ===
        self.representation = Representation(
            num_layers=cfg["num_layers"],
            atomic_numbers=cfg["atomic_numbers"],
            cutoff=cfg["cutoff"],
            avg_num_neighbors=cfg["avg_num_neighbors"],
            magmoms_norm_by_element=cfg["magmoms_norm_by_element"],
            mmax=cfg["mmax"],
            Lmax=cfg["Lmax"],
            lmax=cfg["lmax"],
            mag_Lmax=cfg["mag_Lmax"],
            num_channel=cfg["num_channel"],
            node_embedding=cfg["node_embedding"],
            edge_embedding=cfg["edge_embedding"],
            edge_update=cfg["edge_update"],
            magnetic_edge_update=cfg["magnetic_edge_update"],
            radial_basis=cfg["radial_basis"],
            atomic_basis=cfg["atomic_basis"],
            product_basis=cfg["product_basis"],
            target_irreps=self.target_irreps,
            invariant_property=cfg["invariant_property"],
            equivariant_property=cfg["equivariant_property"],
            universal_embedding=cfg["universal_embedding"],
            resnet=cfg["resnet"],
            layer_norm=cfg["layer_norm"],
            dropout=cfg["dropout"],
            parity=cfg["parity"],
            use_one_body_magmoms=self.use_one_body_magmoms,
        )
        self.use_time_reversal = self.representation.use_time_reversal

        # === Readout ===
        if self.representation.use_dens:
            assert self.use_alllayer == False
        for_scalar_readout = {
            "num_layers": cfg["num_layers"],
            "hidden_channel": cfg["readout_emlp"]["hidden"],
            "bias": cfg["readout_emlp"]["bias"],
            "num_elements": self.num_elements,
            "num_fidelities": len(cfg["fidelity"]),
            "use_alllayer": self.use_alllayer,
            "parity": cfg["parity"],
            "irreps_in": [prod.irreps_out for prod in self.representation.products],
        }
        for_tensor_readout = {
            "num_layers": cfg["num_layers"],
            "hidden_channel": cfg["readout_emlp"]["hidden"],
            "bias": cfg["readout_emlp"]["bias"],
            "num_elements": self.num_elements,
            "num_fidelities": len(cfg["fidelity"]),
            "use_alllayer": self.use_alllayer,
            "parity": cfg["parity"],
            "irreps_in": [prod.irreps_out for prod in self.representation.products],
        }

        # === Energy ===
        if "energy" in self.target_property:
            self.energy_readouts = build_scalar_readout(
                irreps_out="0e", **for_scalar_readout
            )
            if self.use_one_body_magmoms:
                self.one_body_magmoms_readout = build_element_scalar_readout(
                    num_layers=1,
                    hidden_channel=[],
                    bias=False,
                    num_elements=self.num_elements,
                    num_fidelities=len(cfg["fidelity"]),
                    use_alllayer=False,
                    parity=cfg["parity"],
                    irreps_in=[
                        o3.Irreps(f"{self.representation.magnetic_basis.num_basis}x0e")
                    ],
                    irreps_out="0e",
                )[0]
            self.atomic_energy_layer = OneHotToAtomicEnergy(
                cfg["atomic_energies"], cfg["atomic_numbers"]
            )
            if cfg["scale_shift"]["enable"]:
                self.scale_shift = ScaleShift.build_from_config(
                    cfg["statistics"],
                    cfg["scale_shift"],
                    atomic_numbers=cfg["atomic_numbers"],
                )
            # uie base
            if cfg["readout_emlp"]["use_uie"] and len(cfg["invariant_property"]) > 0:
                self.uie_readout = e3nnLinear(
                    f"{cfg['num_channel']}x0e",
                    "1x0e",
                    1,
                    bias=False,
                )

            # === Short range ===
            if cfg["short_range"]["zbl"]["enable"]:
                self.zbl = ZBLBasis(
                    cfg["radial_basis"]["cutoff_fn"],
                    cfg["short_range"]["zbl"]["trainable"],
                    cfg["radial_basis"]["polynomial_cutoff"],
                )

        # === Long range ===
        if les_cfg["enable"]:
            self.les = TACELES(les_arguments, **for_scalar_readout)

        # === Direct Dipolet ===
        if "direct_dipole" in self.target_property:
            self.dipole_readouts = build_tensor_readout(
                irreps_out="1o", **for_tensor_readout
            )

        # === Direct Forces Readout ===
        if "direct_forces" in self.target_property:
            self.direct_forces_readouts = build_tensor_readout(
                irreps_out="1o", **for_tensor_readout
            )

        # === Direct Polarizability ===
        if "direct_polarizability" in self.target_property:
            self.direct_polarizability_readout0s = build_scalar_readout(
                irreps_out="0e", **for_scalar_readout
            )
            self.direct_polarizability_readout2s = build_tensor_readout(
                irreps_out="2e", **for_tensor_readout
            )
            self.direct_polarizability_basis_change = DirectPolarizability()

        # === Direct Virials ===
        if (
            "direct_virials" in self.target_property
            or "direct_stress" in self.target_property
        ):
            self.direct_virials_readout0s = build_scalar_readout(
                irreps_out="0e", **for_scalar_readout
            )
            self.direct_virials_readout2s = build_tensor_readout(
                irreps_out="2e", **for_tensor_readout
            )
            self.direct_virials_basis_change = DirectVirials()

        # === Charges ===
        if "charges" in self.target_property:
            self.predict_charges_method = cfg["special"]["charges"]["method"]
            if self.predict_charges_method == "lagrangian":
                self.chi_readouts = build_scalar_readout(
                    irreps_out="0e", **for_scalar_readout
                )
                self.eta_readouts = build_scalar_readout(
                    irreps_out="0e", **for_scalar_readout
                )
            elif self.predict_charges_method == "uniform_distribution":
                self.charges_readouts = build_scalar_readout(
                    irreps_out="0e", **for_scalar_readout
                )
            else:
                raise ValueError(
                    f"Unknown predict_charges_method: {self.predict_charges_method}. "
                    "Supported methods are ['lagrangian', 'uniform_distribution']."
                )

        # === abs_final_collinear_magmoms ===
        if "abs_final_collinear_magmoms" in self.target_property:
            self.abs_final_collinear_magmoms_readouts = build_scalar_readout(
                irreps_out="0e", **for_scalar_readout
            )  # TODO, check

        # === DeNs noise ===
        if self.representation.use_dens:
            self.dens_noise_readouts = build_tensor_readout(
                irreps_out="1o", **for_tensor_readout
            )

        # self.normalizers = torch.nn.ModuleDict()
        # for p in self.target_property:
        #     self.normalizers[p] = Normalizer(
        #         cfg['normalizer'][p].get('mean', 0.0),
        #         cfg['normalizer'][p].get('rmsd', 1.0),
        #     )

    def readout_fn(
        self,
        data: Dict[str, torch.Tensor],
        graph,
        from_representation: Dict[str, Union[torch.Tensor, None]],
    ) -> Dict[str, Union[torch.Tensor, None]]:

        batch = data["batch"]
        descriptors = from_representation["descriptors"]

        nlocal, _ = graph.lmp_natoms
        num_graphs = graph.num_graphs
        node_fidelity = graph.node_fidelity
        num_atoms_arange = graph.num_atoms_arange
        dtype = data["node_attrs"].dtype
        device = data["node_attrs"].device

        # === Energy ===
        E = None
        e_node = None
        e_base_graph = None
        if "energy" in self.target_property:
            e_base_node = self.atomic_energy_layer(data["node_attrs"])[
                num_atoms_arange, node_fidelity
            ]
            e_base_graph = scatter_sum(e_base_node, batch, dim=-1, dim_size=num_graphs)
            e_list = []
            e_one_body_magmoms_node = None
            for ii, energy_readout in enumerate(self.energy_readouts):
                if not self.use_alllayer:
                    ii = -1
                e_list.append(
                    energy_readout(descriptors[ii], node_fidelity)[
                        num_atoms_arange, node_fidelity
                    ]
                )
            if self.use_one_body_magmoms:
                one_body_magmoms_basis = from_representation["one_body_magmoms_basis"]
                if one_body_magmoms_basis is None:
                    raise RuntimeError("one-body magnetic-moment basis is unavailable")
                e_one_body_magmoms_node = self.one_body_magmoms_readout(
                    one_body_magmoms_basis,
                    node_fidelity=node_fidelity,
                    node_attrs=data["node_attrs"],
                )[num_atoms_arange, node_fidelity]
            e_node = torch.sum(torch.stack(e_list, dim=0), dim=0)
            # === ZBL ===
            if hasattr(self, "zbl"):
                e_zbl_node = self.zbl(
                    graph.edge_length,
                    data["node_attrs"],
                    data["edge_index"],
                    self.atomic_numbers,
                )[num_atoms_arange]
                if self.scale_zbl:
                    e_node = e_node + e_zbl_node
            # === scale and shift ===
            if hasattr(self, "scale_shift"):
                e_node = self.scale_shift(
                    e_node,
                    data["node_attrs"][num_atoms_arange],
                    data["ptr"],
                    data["edge_index"],
                    data["batch"],
                    node_fidelity,
                )
            if hasattr(self, "zbl") and not self.scale_zbl:
                e_node = e_node + e_zbl_node
            if e_one_body_magmoms_node is not None:
                e_node = e_node + e_one_body_magmoms_node
            # === uie ===
            if hasattr(self, "uie_readout"):
                e_uie_node = self.uie_readout(from_representation["uie_feats"])
                e_node = e_node + e_uie_node[num_atoms_arange, 0, 0]
            e_graph = scatter_sum(e_node, batch, dim=-1, dim_size=num_graphs)
            e_node = e_base_node + e_node
            E = e_base_graph + e_graph
            # E = e_graph

        # === Direct Forces ===
        D_F = None
        if "direct_forces" in self.target_property:
            if from_representation["decouple_node_feats1"] is None:
                d_f_list = []
                for ii, direct_forces_readout in enumerate(self.direct_forces_readouts):
                    if not self.use_alllayer:
                        ii = -1
                    d_f_list.append(
                        direct_forces_readout(
                            descriptors[ii],
                            node_fidelity,
                        ).reshape(-1, self.num_fidelities, 3)[
                            num_atoms_arange, node_fidelity, :
                        ]
                    )
                D_F = torch.sum(torch.stack(d_f_list, dim=-1), dim=-1)
            else:
                d_f_list = []
                for ii, direct_forces_readout in enumerate(self.direct_forces_readouts):
                    if not self.use_alllayer:
                        ii = -1
                    d_f_list.append(
                        direct_forces_readout(
                            from_representation["decouple_node_feats1"],
                            node_fidelity,
                        ).reshape(-1, self.num_fidelities, 3)[
                            num_atoms_arange, node_fidelity, :
                        ]
                    )
                D_F = torch.sum(torch.stack(d_f_list, dim=-1), dim=-1)

        # === Direct Dipole ===
        D = None
        if "direct_dipole" in self.target_property:
            d_base = compute_fixed_charge_dipole(
                charges=data["charges"],
                positions=data["positions"],
                batch=data["batch"],
                num_graphs=num_graphs,
            )
            d_list = []
            for ii, dipole_readout in enumerate(self.dipole_readouts):
                if not self.use_alllayer:
                    ii = -1
                d_list.append(
                    dipole_readout(
                        descriptors[ii],
                        node_fidelity,
                    ).reshape(-1, self.num_fidelities, 3)[
                        num_atoms_arange, node_fidelity, :
                    ]
                )
            d_node = torch.sum(torch.stack(d_list, dim=-1), dim=-1)
            d_graph = scatter_sum(d_node, batch, dim=0, dim_size=num_graphs)
            D = d_base + d_graph

        # === Direct Polarizability ===
        ALPHA = None
        if "direct_polarizability" in self.target_property:
            alpha0_list = []
            alpha2_list = []
            for ii, (polarizability_readout0, polarizability_readout2) in enumerate(
                zip(
                    self.direct_polarizability_readout0s,
                    self.direct_polarizability_readout2s,
                )
            ):
                if not self.use_alllayer:
                    ii = -1
                alpha0_list.append(
                    polarizability_readout0(
                        descriptors[ii],
                    )[num_atoms_arange, node_fidelity]
                )
                alpha2_list.append(
                    polarizability_readout2(
                        descriptors[ii],
                        node_fidelity,
                    ).reshape(-1, self.num_fidelities, 5)[
                        num_atoms_arange, node_fidelity, :
                    ]
                )
            alpha0_node = torch.sum(torch.stack(alpha0_list, dim=-1), dim=-1)
            alpha2_node = torch.sum(torch.stack(alpha2_list, dim=-1), dim=-1)
            alpha0_graph = scatter_sum(alpha0_node, batch, dim=0, dim_size=num_graphs)
            alpha2_graph = scatter_sum(alpha2_node, batch, dim=0, dim_size=num_graphs)
            ALPHA = self.direct_polarizability_basis_change(alpha0_graph, alpha2_graph)

        # === Direct Virials and Stress ===
        D_V = None
        D_S = None
        if (
            "direct_virials" in self.target_property
            or "direct_stress" in self.target_property
        ):
            if from_representation["decouple_node_feats1"] is None:
                d_v0_list = []
                d_v2_list = []
                for ii, (direct_virials_readout0, direct_virials_readout2) in enumerate(
                    zip(self.direct_virials_readout0s, self.direct_virials_readout2s)
                ):
                    if not self.use_alllayer:
                        ii = -1
                    d_v0_list.append(
                        direct_virials_readout0(
                            descriptors[ii],
                        )[num_atoms_arange, node_fidelity]
                    )
                    d_v2_list.append(
                        direct_virials_readout2(
                            descriptors[ii],
                        ).reshape(-1, self.num_fidelities, 5)[
                            num_atoms_arange, node_fidelity, :
                        ]
                    )
                d_v0_node = torch.sum(torch.stack(d_v0_list, dim=-1), dim=-1)
                d_v2_node = torch.sum(torch.stack(d_v2_list, dim=-1), dim=-1)
                d_v0_graph = scatter_sum(d_v0_node, batch, dim=0, dim_size=num_graphs)
                d_v2_graph = scatter_sum(d_v2_node, batch, dim=0, dim_size=num_graphs)
                D_V = self.direct_virials_basis_change(d_v0_graph, d_v2_graph)
                VOLUME = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
                D_S = -D_V / VOLUME.view(-1, 1, 1)
                D_S = torch.where(torch.abs(D_S) < 1e10, D_S, torch.zeros_like(D_S))
            else:
                d_v0_list = []
                d_v2_list = []
                for ii, (direct_virials_readout0, direct_virials_readout2) in enumerate(
                    zip(self.direct_virials_readout0s, self.direct_virials_readout2s)
                ):
                    if not self.use_alllayer:
                        ii = -1
                    d_v0_list.append(
                        direct_virials_readout0(
                            from_representation["decouple_node_feats1"],
                        )[num_atoms_arange, node_fidelity]
                    )
                    d_v2_list.append(
                        direct_virials_readout2(
                            from_representation["decouple_node_feats1"],
                        ).reshape(-1, self.num_fidelities, 5)[
                            num_atoms_arange, node_fidelity, :
                        ]
                    )
                d_v0_node = torch.sum(torch.stack(d_v0_list, dim=-1), dim=-1)
                d_v2_node = torch.sum(torch.stack(d_v2_list, dim=-1), dim=-1)
                d_v0_graph = scatter_sum(d_v0_node, batch, dim=0, dim_size=num_graphs)
                d_v2_graph = scatter_sum(d_v2_node, batch, dim=0, dim_size=num_graphs)
                D_V = self.direct_virials_basis_change(d_v0_graph, d_v2_graph)
                VOLUME = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
                D_S = -D_V / VOLUME.view(-1, 1, 1)
                D_S = torch.where(torch.abs(D_S) < 1e10, D_S, torch.zeros_like(D_S))

        # === Charges ===
        CHARGES = None
        if "charges" in self.target_property:
            if self.predict_charges_method == "lagrangian":
                chi_list = []
                eta_list = []
                for ii, (chi_readout, eta_readout) in enumerate(
                    zip(self.chi_readouts, self.eta_readouts)
                ):
                    if not self.use_alllayer:
                        ii = -1
                    chi_list.append(
                        chi_readout(descriptors[ii])[num_atoms_arange, node_fidelity]
                    )
                    eta_list.append(
                        eta_readout(descriptors[ii])[num_atoms_arange, node_fidelity]
                    )
                chi_node = torch.sum(torch.stack(chi_list, dim=-1), dim=-1)
                eta_node = torch.sum(torch.stack(eta_list, dim=-1), dim=-1)
                eta_node = torch.hypot(
                    eta_node, torch.tensor(1e-6, device=device, dtype=dtype)
                )
                eta_node = torch.reciprocal(eta_node)
                lambda_graph = (
                    data["total_charge"]
                    + scatter_sum(
                        chi_node * eta_node, batch, dim=-1, dim_size=num_graphs
                    )
                ) / scatter_sum(eta_node, batch, dim=-1, dim_size=num_graphs)
                lambda_node = lambda_graph[batch]
                CHARGES = lambda_node * (eta_node) - (chi_node * eta_node)
            elif self.predict_charges_method == "uniform_distribution":
                c_list = []
                for ii, charges_readout in enumerate(self.charges_readouts):
                    if not self.use_alllayer:
                        ii = -1
                    c_list.append(
                        charges_readout(descriptors[ii])[
                            num_atoms_arange, node_fidelity
                        ]
                    )
                c_node = torch.sum(torch.stack(c_list, dim=-1), dim=-1)
                c_graph = scatter_sum(c_node, batch, dim=-1, dim_size=num_graphs)
                c_delta_node = (c_graph - data["total_charge"]) / (
                    data["ptr"][1:] - data["ptr"][:-1]
                )
                CHARGES = c_node + c_delta_node[batch]

        # === ABS_F_C_MAG ===
        ABS_F_C_MAG = None
        if "abs_final_collinear_magmoms" in self.target_property:
            mag_list = []
            for ii, abs_final_collinear_magmoms_readout in enumerate(
                self.abs_final_collinear_magmoms_readouts
            ):
                if not self.use_alllayer:
                    ii = -1
                mag_list.append(
                    torch.abs(
                        abs_final_collinear_magmoms_readout(descriptors[ii])[
                            num_atoms_arange, node_fidelity
                        ]
                    )
                )
            ABS_F_C_MAG = torch.sum(torch.stack(mag_list, dim=-1), dim=-1)

        if hasattr(self, "les"):
            les_results = self.les(
                descriptors=descriptors,
                node_fidelity=node_fidelity,
                num_atoms_arange=num_atoms_arange,
                positions=data["positions"],
                lattice=graph.lattice,
                batch=batch,
                atomic_numbers=data["atomic_numbers"],
                external_field=data.get("electric_field"),
                pbc=data.get("pbc"),
            )
            LES_E = les_results["les_energy"]
            if LES_E is None:
                LES_E = torch.zeros_like(E)
            E += LES_E
        else:
            les_results = {
                "les_energy": None,
                "les_latent_charges": None,
                "les_latent_dipoles": None,
                "les_latent_quads": None,
                "les_latent_alphas": None,
                "les_latent_kappas": None,
                "les_born_effective_charges": None,
            }
            LES_E = None

        scalar_descriptor = None
        if "0e" in self.target_irreps:
            scalar_descriptor_list = []
            for descriptor in descriptors:
                scalar_descriptor_list.append(descriptor[:, : self.num_channel])
            scalar_descriptor = torch.cat(scalar_descriptor_list, dim=-1)

        # === DeNS noise ===
        dens_noise = None
        if hasattr(self, "dens_noise_readouts"):
            if from_representation["decouple_node_feats2"] is not None:
                noise_list = []
                for ii, dens_noise_readout in enumerate(self.dens_noise_readouts):
                    if not self.use_alllayer:
                        ii = -1
                    noise_list.append(
                        dens_noise_readout(
                            from_representation["decouple_node_feats2"],
                            node_fidelity,
                        ).reshape(-1, self.num_fidelities, 3)[
                            num_atoms_arange, node_fidelity, :
                        ]
                    )
                dens_noise = torch.sum(torch.stack(noise_list, dim=-1), dim=-1)
            else:
                noise_list = []
                for ii, dens_noise_readout in enumerate(self.dens_noise_readouts):
                    if not self.use_alllayer:
                        ii = -1
                    noise_list.append(
                        dens_noise_readout(
                            descriptors[ii],
                            node_fidelity,
                        ).reshape(-1, self.num_fidelities, 3)[
                            num_atoms_arange, node_fidelity, :
                        ]
                    )
                dens_noise = torch.sum(torch.stack(noise_list, dim=-1), dim=-1)

        return {
            "energy": E,
            "node_energy": e_node,  # not include les
            # "e_base_graph": e_base_graph,
            "direct_dipole": D,
            "direct_polarizability": ALPHA,
            "direct_forces": D_F,
            "direct_virials": D_V,
            "direct_stress": D_S,
            "charges": CHARGES,
            "les_energy": LES_E,
            "les_latent_charges": les_results["les_latent_charges"],
            "les_latent_dipoles": les_results["les_latent_dipoles"],
            "les_latent_quads": les_results["les_latent_quads"],
            "les_latent_alphas": les_results["les_latent_alphas"],
            "les_latent_kappas": les_results["les_latent_kappas"],
            "les_born_effective_charges": les_results["les_born_effective_charges"],
            "scalar_descriptor": scalar_descriptor,
            "abs_final_collinear_magmoms": ABS_F_C_MAG,
            "noise_vec": dens_noise,
        }

    def forward(self, data: Dict[str, torch.Tensor], graph) -> Dict[str, Any]:
        rep = self.representation(data, graph)
        return self.readout_fn(data, graph, rep) | rep
