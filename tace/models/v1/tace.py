################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional, Any


import torch
from torch import nn, Tensor
from cartnn.math import ZBLBasis


from .mlp import MLP
from .layers import OneHotToAtomicEnergy, ScaleShift
from .readout import NodeLinearReadOut, NodeNonLinearReadOut
from .representation import TACEDescriptor
from .utils import (
    Graph,
    RADIAL_BASIS,
    ANGULAR_BASIS,
    RADIAL_MLP,
    INTER,
    PROD,
    ICTD,
    READOUT_MLP,
    SCALE_SHIFT,
    compute_fixed_charge_dipole,
    SHORT_RANGE,
    LONG_RANGE,
)
from ...dataset.statistics import Statistics
from ...utils.torch_scatter import scatter_sum


class TACEV1(torch.nn.Module):
    def __init__(
        self,
        statistics: List[Statistics],
        max_neighbors: int = 999,
        cutoff: float = 5.0,
        max_r_1: int = 2,
        max_r_2: int = 3,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        num_layers: int = 2,
        radial_basis: Dict = RADIAL_BASIS,
        radial_mlp: Dict = RADIAL_MLP,
        angular_basis: Dict = ANGULAR_BASIS,
        ictd: Dict = ICTD,
        inter: Dict = INTER,
        prod: Dict = PROD,
        bias: bool = False,
        readout_mlp: Dict = READOUT_MLP,
        scale_shift: Dict = SCALE_SHIFT,
        target_property: List[str] = ["energy"],
        short_range: Optional[Dict[str, bool]] = None,
        long_range: Optional[Dict[str, bool]] = None,
        embedding_property: Optional[List[str]] = None,
        universal_embedding: Optional[Dict[str, Dict[str, Any]]] = None,
        conservations: Optional[Dict[str, bool]] = None,
        mixed_precision: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(statistics, List):
            statistics = [statistics]
        assert isinstance(radial_basis, Dict), "cfg.model.config.radial_basis must be a Dict"
        assert isinstance(radial_mlp, Dict), "cfg.model.config.radial_mlp must be a Dict"
        assert isinstance(angular_basis, Dict), "cfg.model.config.angular_basis must be a Dict"
        assert isinstance(ictd, Dict), "cfg.model.config.ictd must be a Dict"
        assert isinstance(inter, Dict), "cfg.model.config.inter must be a Dict"
        assert isinstance(prod, Dict), "cfg.model.config.prod must be a Dict"
        assert isinstance(readout_mlp, Dict), "cfg.model.config.readout_mlp must be a Dict"
        assert isinstance(scale_shift, Dict), "cfg.model.config.scale_shift must be a Dict"
        assert short_range is None or isinstance(short_range, Dict), "cfg.model.config.short_range must be a Dict or None"
        assert long_range is None or isinstance(long_range, Dict), "cfg.model.config.long_range must be a Dict or None"
        assert embedding_property is None or isinstance(embedding_property, List), "embedding_property must be a List or None"
        assert conservations is None or isinstance(conservations, Dict), "cfg.model.config.conservations must be a Dict or None"
        assert universal_embedding is None or isinstance(universal_embedding, Dict), "cfg.model.config.universal_embedding must be a Dict or None"
        assert mixed_precision is None or isinstance(mixed_precision, Dict), "cfg.model.config.mixed_precision must be a Dict or None"

        # === init ===
        atomic_numbers = sorted(statistics[0]['atomic_numbers'])
        if "energy" in target_property: 
            atomic_energies = [stats['atomic_energy'] for stats in statistics]
        else:
            atomic_energies = None
        self.target_property = target_property
        self.embedding_property = embedding_property or []
        self.universal_embedding = universal_embedding or {}
        self.conservations = conservations or {}
        self.mixed_precision = mixed_precision or {}
        self.register_buffer('atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype()))
        self.register_buffer('num_layers', torch.tensor(num_layers, dtype=torch.int64))
        if isinstance(max_neighbors, int): self.register_buffer('max_neighbors', torch.tensor(max_neighbors, dtype=torch.int64))

            # universal embedding
        invariant_embedding_property = self.universal_embedding.get("invariant", {})
        equivariant_embedding_property = self.universal_embedding.get("equivariant", {})
        self.enable_electric_field = "electric_field" in equivariant_embedding_property
        self.enable_magnetic_field = "magnetic_field" in equivariant_embedding_property

            # conservation property
        self.enable_Qeq = self.conservations.get("charges", {}).get('enable_Qeq', False)
        self.enable_Quni = self.conservations.get("charges", {}).get("enable_Quni", False)

            # readout
        act = readout_mlp.get('act', "silu")
        gate = readout_mlp.get('gate', "silu")
        readout_bias = readout_mlp.get('bias', False)
        enable_uie_readout = readout_mlp.get('enable_uie', True)
        enable_multi_head = self.mixed_precision.get('enable_multi_head', False)
        hidden_dim = readout_mlp.get('hidden', [16]) or [16]
        self.level_names = self.mixed_precision.get('level_names', ['default']) or ['default']
        num_levels = len(self.level_names)
        num_levels_for_readout = 1
        if enable_multi_head:
            hidden_dim = [tmp * num_levels for tmp in hidden_dim]
            num_levels_for_readout = num_levels
        self.enable_multi_head = (enable_multi_head) and (num_levels > 1) and len(hidden_dim) > 0
        self.enable_nonlinearty_for_tensor = readout_mlp.get('enable_nonlinearty_for_tensor', False)
        self.use_only_last_layer = readout_mlp.get('use_only_last_layer', False)

            # short range and long range correlation
        short_range = short_range or SHORT_RANGE
        long_range = long_range or LONG_RANGE
        zbl = short_range.get('enable_zbl', False)
        les = long_range.get('les', LONG_RANGE['les'])
        ictd_weight = ictd.get('weight', 'max')

        # === Representation ===
        for_descriptor = {
            "cutoff": cutoff,
            "num_layers": num_layers,
            "max_r_1": max_r_1,
            "max_r_2": max_r_2,
            "num_channel": num_channel,
            "num_channel_hidden": num_channel_hidden,
            "radial_basis": radial_basis,
            "radial_mlp": radial_mlp,
            "angular_basis": angular_basis,
            "ictd": ictd,
            "inter": inter,
            "prod": prod,
            "bias": bias,
            "statistics": statistics,
            "target_property": target_property,
            "universal_embedding": universal_embedding,
            "readout_mlp": readout_mlp,
        }
        self.descriptor = TACEDescriptor(**for_descriptor)

        self.precompute = prod.get("precompute", True)

        # === Energy ReadOut ===
        if "energy" in self.target_property:
            energy_readouts = [
                MLP(
                    in_dim=num_channel,
                    out_dim=num_levels_for_readout,
                    hidden_dim=hidden_dim if i == num_layers - 1 else [],
                    act=act if i == num_layers - 1 else None,
                    bias=readout_bias,
                    forward_weight_init=False if i == num_layers - 1 else True,
                    num_levels=num_levels,
                    enable_multi_head=enable_multi_head,
                ) for i in range(num_layers)
            ]
            if self.use_only_last_layer:
                self.energy_readouts = nn.ModuleList([energy_readouts[-1]])
            else:
                self.energy_readouts = nn.ModuleList(energy_readouts)
            del energy_readouts
            if zbl:
                self.zbl = ZBLBasis(radial_basis["polynomial_cutoff"])
            self.atomic_energy_layer = OneHotToAtomicEnergy(atomic_energies)
            self.scale_shift = ScaleShift.build_from_config(statistics, scale_shift)

        if "charges" in self.target_property:
            if self.enable_Qeq:
                # === Qeq Method ReadOut ===
                chi_readouts = [
                    MLP(
                        in_dim=num_channel,
                        out_dim=num_levels_for_readout,
                        hidden_dim=hidden_dim if i == num_layers - 1 else [],
                        act=act if i == num_layers - 1 else None,
                        bias=readout_bias,
                        forward_weight_init=False if i == num_layers - 1 else True,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    ) for i in range(num_layers)
                ]
            
                eta_readouts = [
                    MLP(
                        in_dim=num_channel,
                        out_dim=num_levels_for_readout,
                        hidden_dim=hidden_dim if i == num_layers - 1 else [],
                        act=act if i == num_layers - 1 else None,
                        bias=readout_bias,
                        forward_weight_init=False if i == num_layers - 1 else True,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    ) for i in range(num_layers)
                ]
                if self.use_only_last_layer:
                    self.chi_readouts = nn.ModuleList([chi_readouts[-1]])
                    self.eta_readouts = nn.ModuleList([eta_readouts[-1]])
                else:
                    self.chi_readouts = nn.ModuleList(chi_readouts)
                    self.eta_readouts = nn.ModuleList(eta_readouts)
                del chi_readouts, eta_readouts
            else:
                # === Uniform Method ReadOut ===
                charges_readouts = [
                    MLP(
                        in_dim=num_channel,
                        out_dim=num_levels_for_readout,
                        hidden_dim=hidden_dim if i == num_layers - 1 else [],
                        act=act if i == num_layers - 1 else None,
                        bias=readout_bias,
                        forward_weight_init=False if i == num_layers - 1 else True,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    ) for i in range(num_layers)
                ]

                if self.use_only_last_layer:
                    self.charges_readouts = nn.ModuleList([charges_readouts[-1]])
                else:
                    self.charges_readouts = nn.ModuleList(charges_readouts)
                del chi_readouts, eta_readouts

        # === Universal Invariant Embedding ReadOut ===
        if universal_embedding is not None:
            if len(invariant_embedding_property) > 0 and enable_uie_readout:
                self.uie_readout = MLP(
                    num_channel,
                    1,
                    hidden_dim=[],
                    act=None,
                    bias=False,
                    forward_weight_init=True,
                )

        # === Direct Dipole ReadOut ===
        if "direct_dipole" in self.target_property:
            if self.enable_nonlinearty_for_tensor:
                dipole_readouts = []
                for _ in range(num_layers-1):
                    dipole_readouts.append(
                        NodeLinearReadOut(
                            in_dim=num_channel,
                            out_dim=num_levels,
                            bias=False,  # can not be True
                            atomic_numbers=None,
                        ) 
                    )
                dipole_readouts.append(
                    NodeNonLinearReadOut(
                        in_dim=num_channel,
                        hidden_dim=hidden_dim,
                        out_dim=num_levels,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                        gate=gate,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    )         
                )
            else:
                dipole_readouts = [
                    NodeLinearReadOut(
                        in_dim=num_channel,
                        out_dim=num_levels,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                    ) for _ in range(num_layers)
                ]
            if self.use_only_last_layer:
                self.dipole_readouts = nn.ModuleList([dipole_readouts[-1]])
            else:
                self.dipole_readouts = nn.ModuleList(dipole_readouts)
            del dipole_readouts
            
        # === Direct Dipole ReadOut ===
        if "direct_forces" in self.target_property:
            if self.enable_nonlinearty_for_tensor:
                direct_forces_readouts = []
                for _ in range(num_layers-1):
                    direct_forces_readouts.append(
                        NodeLinearReadOut(
                            in_dim=num_channel,
                            out_dim=num_levels,
                            bias=False,  # can not be True
                            atomic_numbers=None,
                        ) 
                    )
                direct_forces_readouts.append(
                    NodeNonLinearReadOut(
                        in_dim=num_channel,
                        hidden_dim=hidden_dim,
                        out_dim=num_levels,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                        gate=gate,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    )         
                )
            else:
                direct_forces_readouts = [
                    NodeLinearReadOut(
                        in_dim=num_channel,
                        out_dim=num_levels,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                    ) for _ in range(num_layers)
                ]
            if self.use_only_last_layer:
                self.direct_forces_readouts = nn.ModuleList([direct_forces_readouts[-1]])
            else:
                self.direct_forces_readouts = nn.ModuleList(direct_forces_readouts)
            del direct_forces_readouts

        # === Direct Polarizability ReadOut ===
        if "direct_polarizability" in self.target_property:
            self.manual_symmetrize = (ictd_weight != "max")
            polarizability_readout0s = [
                MLP(
                    in_dim=num_channel,
                    out_dim=num_levels_for_readout,
                    hidden_dim=hidden_dim if i == num_layers - 1 else [],
                    act=act if i == num_layers - 1 else None,
                    bias=readout_bias,
                    forward_weight_init=False if i == num_layers - 1 else True,
                    num_levels=num_levels,
                    enable_multi_head=enable_multi_head,
                ) for i in range(num_layers)
            ]
            if self.enable_nonlinearty_for_tensor:
                polarizability_readout2s = []
                for _ in range(num_layers-1):
                    polarizability_readout2s.append(
                        NodeLinearReadOut(
                            in_dim=num_channel,
                            out_dim=1,
                            bias=False,  # can not be True
                            atomic_numbers=None,
                        ) 
                    )
                polarizability_readout2s.append(
                    NodeNonLinearReadOut(
                        in_dim=num_channel,
                        hidden_dim=hidden_dim,
                        out_dim=1,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                        gate=gate,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    )         
                )
            else:
                polarizability_readout2s = [
                    NodeLinearReadOut(
                        in_dim=num_channel,
                        out_dim=1,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                    ) for _ in range(num_layers)
                ]
            if self.use_only_last_layer:
                self.polarizability_readout0s = nn.ModuleList([polarizability_readout0s[-1]])
                self.polarizability_readout2s = nn.ModuleList([polarizability_readout2s[-1]])
            else:
                self.polarizability_readout0s = nn.ModuleList(polarizability_readout0s)
                self.polarizability_readout2s = nn.ModuleList(polarizability_readout2s)
            del polarizability_readout0s, polarizability_readout2s
            
        # === Direct Virials Stress ReadOut ===
        if 'direct_virials' in self.target_property or 'direct_stress' in self.target_property:
            self.manual_symmetrize = (ictd_weight != "max")
            direct_virials_readout0s = [
                MLP(
                    in_dim=num_channel,
                    out_dim=num_levels_for_readout,
                    hidden_dim=hidden_dim if i == num_layers - 1 else [],
                    act=act if i == num_layers - 1 else None,
                    bias=readout_bias,
                    forward_weight_init=False if i == num_layers - 1 else True,
                    num_levels=num_levels,
                    enable_multi_head=enable_multi_head,
                ) for i in range(num_layers)
            ]

            if self.enable_nonlinearty_for_tensor:
                direct_virials_readout2s = []
                for _ in range(num_layers-1):
                    direct_virials_readout2s.append(
                        NodeLinearReadOut(
                            in_dim=num_channel,
                            out_dim=1,
                            bias=False,  # can not be True
                            atomic_numbers=None,
                        ) 
                    )
                direct_virials_readout2s.append(
                    NodeNonLinearReadOut(
                        in_dim=num_channel,
                        hidden_dim=hidden_dim,
                        out_dim=1,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                        gate=gate,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    )         
                )
            else:
                direct_virials_readout2s = [
                    NodeLinearReadOut(
                        in_dim=num_channel,
                        out_dim=1,
                        bias=False,  # can not be True
                        atomic_numbers=None,
                    ) for _ in range(num_layers)
                ]
            if self.use_only_last_layer:
                self.direct_virials_readout0s = nn.ModuleList([direct_virials_readout0s[-1]])
                self.direct_virials_readout2s = nn.ModuleList([direct_virials_readout2s[-1]])
            else:
                self.direct_virials_readout0s = nn.ModuleList(direct_virials_readout0s)
                self.direct_virials_readout2s = nn.ModuleList(direct_virials_readout2s)
            del direct_virials_readout0s, direct_virials_readout2s

        # === long range correction - les ===
        if les is not None:
            assert isinstance(
                les, Dict
            ), "cfg.model.config.les should be None or Dict"
            if les.get("enable_les", False):
                # === les init ===
                try:
                    from les import Les
                except ImportError as e:
                    raise ImportError(
                        "Can not import ``les``(Latent Ewald Summation Library). Please install the 'les' library from https://github.com/ChengUCB/les."
                    ) from e
                if "les_arguments" in les:
                    assert les is None or isinstance(
                        les, Dict
                    ), "cfg.model.config.les.les_arguments should be None or Dict"
                    les_arguments = les["les_arguments"]
                    if les_arguments is None:
                        les_arguments = {"use_atomwise": False}
                self.compute_bec = les_arguments.get("compute_bec", False)
                self.bec_output_index = les_arguments.get("bec_output_index", None)

                # === les module ===
                self.les = Les(les_arguments=les_arguments)
                les_readouts = [
                    MLP(
                        in_dim=num_channel,
                        out_dim=num_levels_for_readout,
                        hidden_dim=hidden_dim if i == num_layers - 1 else [],
                        act=act if i == num_layers - 1 else None,
                        bias=readout_bias,
                        forward_weight_init=False if i == num_layers - 1 else True,
                        num_levels=num_levels,
                        enable_multi_head=enable_multi_head,
                    ) for i in range(num_layers)
                ] 
                if self.use_only_last_layer:
                    self.les_readouts = nn.ModuleList([les_readouts[-1]])
                else:
                    self.les_readouts = nn.ModuleList(les_readouts)
                del les_readouts

    def readout_fn(
        self,
        data: Dict[str, Tensor],
        graph: Graph,
        from_representation: Dict[str, Optional[Tensor]]
    ) -> Dict[str, Optional[Tensor]]:

        batch = data["batch"]
        node_attrs = data['node_attrs']
        descriptors = from_representation['descriptors']

        nlocal, _ = graph.lmp_natoms
        num_graphs = graph.num_graphs
        node_level = graph.node_level
        num_atoms_arange = graph.num_atoms_arange
        dtype = node_attrs.dtype
        device = node_attrs.device

        # === Energy ReadOut ===
        E = None
        node_energy = None
        if "energy" in self.target_property:
            # E0
            e0_node_energy = self.atomic_energy_layer(node_attrs)[num_atoms_arange, node_level]
            E_0 = scatter_sum(src=e0_node_energy, index=batch, dim=-1, dim_size=num_graphs)
            # E1...E2...En
            en_node_energy = []
            for ii, energy_readout in enumerate(self.energy_readouts):
                if self.use_only_last_layer:
                    ii = -1
                if self.enable_multi_head:
                    en_node_energy.append(energy_readout(descriptors[ii][0], node_level)[num_atoms_arange, node_level])  
                else:
                    en_node_energy.append(energy_readout(descriptors[ii][0], node_level).squeeze(-1)[num_atoms_arange])   
            node_energy = torch.sum(torch.stack(en_node_energy, dim=0), dim=0)
            # === ZBL === 
            if hasattr(self, "zbl"):
                pair_node_energy = self.zbl(
                    graph.edge_length, 
                    node_attrs, 
                    data["edge_index"],
                    self.atomic_numbers
                )[num_atoms_arange]
                node_energy = node_energy + pair_node_energy

            # === scale and shift ===
            node_energy = self.scale_shift(
                node_energy, 
                node_attrs[num_atoms_arange], 
                data['ptr'], 
                data['edge_index'], 
                data['batch'],
                node_level,
            )    
            
            # === uie === not support lmp now
            if hasattr(self, "uie_readout"):
                uie_node_energy = self.uie_readout(from_representation['uie_feats'])
                node_energy = node_energy + uie_node_energy.squeeze(-1) 

            E_N = scatter_sum(src=node_energy, index=batch, dim=-1, dim_size=num_graphs)
            node_energy = e0_node_energy + node_energy
            E = E_0 + E_N

        # === Dipole ReadOut ===
        D = None
        if 'direct_dipole' in self.target_property:
            D_0 = compute_fixed_charge_dipole(
                charges=data["charges"],
                positions=data["positions"],
                batch=data["batch"],
                num_graphs=num_graphs,
            )
            dn_node_dipole = []
            for ii, dipole_readout in enumerate(self.dipole_readouts):
                if self.use_only_last_layer:
                    ii = -1
                if self.enable_nonlinearty_for_tensor:
                    dn_node_dipole.append(
                        dipole_readout(
                            descriptors[ii][1],
                            descriptors[ii][0],
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :]
                    )
                else:
                    dn_node_dipole.append(
                        dipole_readout(
                            descriptors[ii][1],
                            None,
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :]
                    )
            node_dipole = torch.sum(torch.stack(dn_node_dipole, dim=-1), dim=-1)
            D_N = scatter_sum(src=node_dipole, index=batch, dim=0, dim_size=num_graphs)
            D = D_0 + D_N

        # === Polarizability ReadOut ===
        ALPHA = None
        if 'direct_polarizability' in self.target_property:
            ALPHA0_N = []
            ALPHA2_N = []
            for ii, (polarizability_readout0, polarizability_readout2) in enumerate(
                zip(self.polarizability_readout0s, self.polarizability_readout2s)
            ):
                if self.use_only_last_layer:
                    ii = -1
                ALPHA0_N.append(
                    polarizability_readout0(
                        descriptors[ii][0],
                    )[num_atoms_arange, node_level]
                )
                if self.enable_nonlinearty_for_tensor:
                    ALPHA2_N.append(
                        dipole_readout(
                            descriptors[ii][2],
                            descriptors[ii][0],
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :, :]
                    )
                else:
                    ALPHA2_N.append(
                        dipole_readout(
                            descriptors[ii][2],
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :, :]
                    )
            ALPHA0_NODE = torch.sum(torch.stack(ALPHA0_N, dim=-1), dim=-1)
            ALPHA2_NODE = torch.sum(torch.stack(ALPHA2_N, dim=-1), dim=-1)
            ALPHA0 = scatter_sum(
                source=ALPHA0_NODE, index=batch, dim=0, dim_size=num_graphs
            )
            ALPHA2 = scatter_sum(src=ALPHA2_NODE, index=batch, dim=0, dim_size=num_graphs)
            I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            ALPHA = ALPHA2 * 0.5 + (ALPHA0 / 3.0).view(-1, 1, 1) * I
            if self.manual_symmetrize:
                ALPHA = ALPHA + ALPHA.permute(0, 2, 1).contiguous()  # only for reducible

        # === Direct Forces ReadOut ===
        D_F = None
        if 'direct_forces' in self.target_property:
            D_F_N = []
            for ii, direct_forces_readout in enumerate(self.direct_forces_readouts):
                if self.use_only_last_layer:
                    ii = -1
                if self.enable_nonlinearty_for_tensor:
                    D_F_N.append(
                        direct_forces_readout(
                            descriptors[ii][1],
                            descriptors[ii][0],
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :]
                    )
                else:
                    D_F_N.append(
                        direct_forces_readout(
                            descriptors[ii][1],
                            None,
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :]
                    )
            D_F = torch.sum(torch.stack(D_F_N, dim=-1), dim=-1)
  
        # === Direct Virials Stress ReadOut ===
        D_V = None
        D_S = None
        if 'direct_virials' in self.target_property or 'direct_stress' in self.target_property:
            d_v0n_node = []
            d_v2n_node = []
            for ii, (direct_virials_readout0, direct_virials_readout2) in enumerate(
                zip(self.direct_virials_readout0s, self.direct_virials_readout2s)
            ):
                if self.use_only_last_layer:
                    ii = -1
                d_v0n_node.append(
                    direct_virials_readout0(
                        descriptors[ii][0],
                    )[num_atoms_arange, node_level]
                )
                d_v2n_node.append(
                    direct_virials_readout2(
                        descriptors[ii][2],
                    )[num_atoms_arange, node_level, :, :]
                )
            d_v0_node = torch.sum(torch.stack(d_v0n_node, dim=-1), dim=-1)
            d_v2_node = torch.sum(torch.stack(d_v2n_node, dim=-1), dim=-1)
            D_V0_N = scatter_sum(
                source=d_v0_node, index=batch, dim=0, dim_size=num_graphs
            )
            D_V2_N = scatter_sum(src=d_v2_node, index=batch, dim=0, dim_size=num_graphs)
            I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            D_V_N = D_V2_N + (D_V0_N / 3.0).view(-1, 1, 1) * I
            D_V = D_V_N
            if self.manual_symmetrize:
                D_V = D_V + D_V.permute(0, 2, 1).contiguous()  # only for reducible
            VOLUME = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
            D_S = -D_V / VOLUME.view(-1, 1, 1)
            D_S = torch.where(torch.abs(D_S) < 1e10, D_S, torch.zeros_like(D_S))

        # === charges predict with conservation === not support lmp
        CHARGES = None
        if self.enable_Qeq:
            if hasattr(self, "chi_readouts"):
                CHI_N = []; ETA_N = []
                for ii, (chi_readout, eta_readout) in enumerate(
                    zip(self.chi_readouts, self.eta_readouts)
                ):
                    if self.use_only_last_layer:
                        ii = -1
                    CHI_N.append(chi_readout(descriptors[ii][0])[num_atoms_arange, node_level])
                    ETA_N.append(eta_readout(descriptors[ii][0])[num_atoms_arange, node_level])
                CHI_N = torch.sum(torch.stack(CHI_N, dim=-1), dim=-1)
                ETA_N = torch.sum(torch.stack(ETA_N, dim=-1), dim=-1)
                ETA_N = torch.hypot(
                    ETA_N, torch.tensor(1e-6, device=device, dtype=dtype)
                )
                INV_ETA_N = torch.reciprocal(ETA_N)
                LAMBDA = (
                    data["total_charge"]
                    + scatter_sum(
                        CHI_N * INV_ETA_N, index=batch, dim=-1, dim_size=num_graphs
                    )
                ) / scatter_sum(INV_ETA_N, index=batch, dim=-1, dim_size=num_graphs)
                node_lambda = LAMBDA[batch]
                CHARGES = node_lambda * (INV_ETA_N) - (CHI_N * INV_ETA_N)
        elif self.enable_Quni:
            if hasattr(self, "charges_readouts"):
                CHARGES_N = []
                for ii, charges_readout in enumerate(self.charges_readouts):
                    if self.use_only_last_layer:
                        ii = -1
                    CHARGES_N.append(charges_readout(descriptors[ii][0])[num_atoms_arange, node_level])
                CHARGES = torch.sum(torch.stack(CHARGES_N, dim=-1), dim=-1)
                PRED_TC = scatter_sum(
                    source=CHARGES, index=batch, dim=-1, dim_size=num_graphs
                )
                LABEL_TC = data["total_charge"]
                delta_per_atom = (LABEL_TC - PRED_TC) / (data["ptr"][1:] - data["ptr"][:-1])
                CHARGES = CHARGES + delta_per_atom[batch]

        if hasattr(self, 'les'): # not support lmp
            LES_LQ = []
            for ii, les_readout in enumerate(self.les_readouts):
                if self.use_only_last_layer:
                    ii = -1
                LES_LQ.append(les_readout(descriptors[ii][0])[num_atoms_arange, node_level])
            LES_LQ = torch.sum(torch.stack(LES_LQ, dim=0), dim=0)  # latent_charges
            LES_RESULTS = self.les(
                latent_charges=LES_LQ,
                positions=graph.positions,
                cell=graph.lattice.view(-1, 3, 3),
                batch=batch,
                compute_energy=True,
                compute_bec=self.compute_bec,
                bec_output_index=self.bec_output_index,
            )
            LES_E = LES_RESULTS["E_lr"]
            LES_BEC = LES_RESULTS["BEC"]
            if LES_E is None:
                LES_E = torch.zeros_like(E)
            else:
                LES_E = LES_E
            E += LES_E
        else:
            LES_LQ = None
            LES_E = None
            LES_BEC = None

        return {
            "energy": E,
            "node_energy": node_energy,
            "direct_dipole": D,
            "direct_polarizability": ALPHA,
            "charges": CHARGES,
            "les_energy": LES_E,
            "les_latent_charges": LES_LQ,
            "les_born_effective_charges": LES_BEC,
            "descriptors": descriptors,
            "direct_forces": D_F,
            "direct_virials": D_V,
            "direct_stress": D_S,
        }
    
    def forward(self, data: Dict[str, Tensor], graph: Graph) -> Dict[str, Any]:
        outs = self.descriptor(data, graph)
        return self.readout_fn(data, graph, outs)

