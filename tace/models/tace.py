################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional, Any


import torch
from torch import nn, Tensor
from torch.jit import ignore


from .mlp import MLP
from .layers import OneHotToAtomicEnergy, ScaleShift
from .radial import ZBLBasis
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
from ..dataset.statistics import Statistics
from ..utils.torch_scatter import scatter_sum


class TACE(torch.nn.Module):
    """
    This is the core model class of TACE.
    For development convenience, all physical property predictions are currently centralized in this class.
    However, this design leads to increased complexity during model compilation.
    Therefore, this version does not provide any acceleration.

    Parameters
    ----------

    statistics : Statistics
        Statistical information about physical quantities, used for normalization, scaling, shifting, etc.
        This object is automatically computed from the training set and does not need to be provided manually.

    target_property : list[str]
        List of target properties to predict.
        This argument should be given in cfg.loss.loss_property, for example: [energy, forces, stress].

    cutoff : float
        Cutoff distance to determine neighbor inclusion.
        This parameter should be provided in YAML file ``misc.cutoff``.

    max_neighbors : int or None
        Maximum number of neighbor atoms considered.
        This parameter should be provided in YAML file ``misc.max_neighbors``.

    max_r_1 : int
        Maximum rank for node features.
        The recommended values are ``0, 1, 2``.

    max_r_2 : int
        Maximum rank for edge attributes (angular).
        The recommended value is ``3``, can set this to ``2`` for speed.

    num_channel : int
        Number of embedding channels.
        The recommended value is ``64``, can set this to ``48`` for less trainable parameters.

    num_channel_hidden : int
        Number of channels in hidden layers.
        The recommended value is set this equal to ``num_channel`` 
        This value is generally expected to be less than or equal to `num_channel`.

    num_layers : int
        Number of layers in the TACE model.
        This determines the ``receptive field``.
        The effective receptive field is ``num_layers × cutoff``.
        The recommended value is ``2``.
        If correlation>=3, always set num_layers = 2.
        If correlation=2, you could enlarge this to 3.
        If correlation=1, you could enlarge this to 3-5. (NequIP, BotNet-like Arch)

    radial_basis : dict
        Configuration for the radial basis functions.

        For example: ::

            {
                "radial_basis": "j0",
                "num_radial_basis": 8,
                "distance_transform": None,
                "polynomial_cutoff": 5,
                "order": 0,
                "trainable": False,
                "apply_cutoff": true,
            }

        Dictionary keys :

            - **radial_basis** : str
                Type of radial basis function. Common options include ``'j0'`` and ``'jn'``.
                See the source code for all available types. If ``'j0'` is used, the ``order`` parameter is ignored.
                If ``'jn'`` is used, you must manually specify the Spherical Bessel function order via ``order``.

            - **num_radial_basis** : int
                Number of radial basis functions. Typically between 8 and 10.

            - **distance_transform** : str or None
                Optional distance transformation applied before basis evaluation.
                If you do not konwo what it means, always set it to None.
                Distance_transform are copy from MACE.
                The available options are ``'Agnesi'``, ``'Soft'``, ``'None'``.
                For morde details, see paper <ACEpotentials.jl: A Julia implementation of the atomic cluster expansion>

            - **polynomial_cutoff** : int or float
                Controls the decay behavior of the cutoff function. A typical value is around 5.

            - **order** : int
                Order of the Spherical Bessel function, only used when ``radial_basis=jn``.

            - **trainable** : bool
                Whether the zeros of the Spherical Bessel function functions are trainable. Normally set to ``False`` to preserve orthogonality.

            - **apply_cutoff** : bool
                Whether use cutoff in radil basis or contraction, true is in radial basis


    radial_mlp : dict
        MLP architecture applied to radial basis features.

        For example: ::

            {
                "hidden": [[64, 64, 64], [64, 64, 64]],
                "act": "silu",
                "bias": false,
                "enable_layer_norm": false,

            }

        Dictionary keys:

            - **hidden** : list[list[int]]
                A list of hidden layer configurations for each MLP layer.
                The outer list must match the number of layers specified by the ``num_layers`` parameter.
                Each inner list defines the hidden dimensions for one MLP.

            - **act** : str
                The activation function applied to each hidden layer.
                Recommended default is ``"silu"``.
                The following options are available:

                    - ``"identity"`` : ``torch.nn.Identity``
                    - ``"relu"`` : ``torch.nn.ReLU``
                    - ``"leaky_relu"`` : ``torch.nn.LeakyReLU``
                    - ``"prelu"`` : ``torch.nn.PReLU``
                    - ``"elu"`` : ``torch.nn.ELU``
                    - ``"selu"`` : ``torch.nn.SELU``
                    - ``"gelu"`` : ``torch.nn.GELU``
                    - ``"silu"`` : ``torch.nn.SiLU``
                    - ``"mish"`` : ``torch.nn.Mish``
                    - ``"softplus"`` : ``torch.nn.Softplus``
                    - ``"softsign"`` : ``torch.nn.Softsign``
                    - ``"tanh"`` : ``torch.nn.Tanh``
                    - ``"sigmoid"`` : ``torch.nn.Sigmoid``
                    - ``"hardtanh"`` : ``torch.nn.Hardtanh``
                    - ``"hardswish"`` : ``torch.nn.Hardswish``
                    - ``"hardsigmoid"`` : ``torch.nn.Hardsigmoid``
                    - ``"tanhshrink"`` : ``torch.nn.Tanhshrink``

            - **bias** : bool

            - **enable_layer_norm** : bool
                If inter.add_source_target_embedding = True else False

    angular_basis: dict
        Configuration for angular basis functions.

        For example: ::

            {
                "type": "ictd",
                "norm": True,
            }

        The structure here should not be altered in any way.

        Dictionary keys:

            - **type** : str
                The type of angular basis functions to use.
                ``"ictd"`` is the currently supported option.

            - **norm** : bool
                Determines whether to multiply the angular basis by a constant.
                **Recommended to set to ``True``.**

    ictd : dict, optional
        Configuration for ICTD scheme.

        For example: ::

            {
                'weight': 'max',
            }

        Available options for the 'weight' key are:

        - ``max``: Only the maximum weight in the given rank is used.
        - ``all``: All different weights are combined using imperfect maner.
        - ``null``: Uses reducible Cartesian tensor features instead.

        We recommend using ``max``.

    inter : dict
        Configuration for the interaction layer.

        For example: ::

            {
                "max_paths": 1,
                "restriction": [None, None],
                "allow_nosym": True,
                "kernel": einsum,
                "residual": False,
                "add_source_target_embedding": false # cat extra source_target embedding to edge_feats
                "normalizer": 
                    {
                        "type": fixed # fixaed or dynamic
                        "hidden": [64]
                        "act_1": 'silu'
                        "act_2": 'tanh'
                        "bias": false
                        "scale_shift_trainable": true # only useful for ``dynamic``
                    },
            }

        Dictionary keys:

            - **max_paths** : int
            Maximum number of paths for each combination.
            If ``ictd['weight'] = 'max'``, then ``max_paths = 1`` is recommended.

            - **restriction** : list[optional[str]]
            Restriction rules on combinations, such as ``r₁ <= r₂``.
            Available options: ``'<=', null``.

            - **allow_nosym** : bool
            If ``True``, allows non-symmetric paths.

            - **kernel** : str
            Use einsum to excute tensor contration, this is the default recommended value. 
            For other kernel, see code by yourself.

            - **residual** : bool
            If ``True``, adds residual connections to the first interaction layers.
            We recommend ``False``.

            - **add_source_target_embedding** : bool
            Not recommended for normal users, see code by yourself.

            - **normalizer** : dict
            set type = fixed
            Other options, are not recommended for normal users, see code by yourself.

    prod : dict
        Product basis layer configuration.

        For example: ::

            {
                "restriction": 
                  {
                    r_1_r_2: <=  
                    r_o_r_1: null 
                  },
                "correlation": 3,
                "allow_nosym": True,
                "element": True,
                "coupled": True,
                "kernel": False,
            }

        Dictionary keys:

            - **restriction** : list[optional[str]]
            Restricts on combinations.
                restriction:
                    r_1_r_2: `null` or `<=`
                    r_o_r_1: `null` or ``<=` or `==`

            - **correlation** : int
            It equals `body-order = correlation + 1` in one layer. We recommend always `3`.

            - **allow_nosym** : bool
            If ``True``, allows non-symmetric paths.

            - **kernel** : str
            Use einsum to excute tensor contration, this is the default recommended value. 
            For other kernel, see code by yourself.

            - **element** : bool
            Whether each element does not share trainable weights for MLP. We recommend always `True`.

            - **coupled** : bool
            Whether to use coupled features. We recommend always `True`.

    short_range : dict

        For example: ::

            {
                'enable_zbl': True
            }
        Whether to use the ZBL repulsive potential in the model.
        Enabling this may slightly increase errors in metrics such as RMSE,
        but it helps improve the stability of molecular dynamics simulations
        and ensures the physical plausibility of generated trajectories.

    bias : bool
        Whether to include bias terms in layers.
        The effect of this parameter is a bit subtle, but in most cases it can be ignored

    readout_mlp : dict
        MLP configuration for the readout layers.

        For example: ::

            {
                "hidden": [16],
                "act": "silu",
            }

        Dictionary keys:

            - **hidden** : list[list[int]]
            A list of hidden layer configurations for each MLP layer.
            The outer list must match the number of layers specified by the ``num_layers`` parameter.
            Each inner list defines the hidden dimensions for one MLP.

            - **act** : str
            The activation function applied to each hidden layer.
            Recommended default is ``"silu"``.
            The following options are available:

            - **gate** : str
            The gate function applied to each hidden layer.
            This parameter is currently **not used** and can be ignored.

    scale_shift : dict
        Normalization and shifting strategy.

        For example: ::

            {
                "scale_type": "rms_forces",
                "shift_type": "mean_delta_energy_per_atom",
                "scale_trainable": False,
                "shift_trainable": False,
                "scale_dict": "auto",
                "shift_dict": "auto",
            }

        Dictionary keys:

            - **scale_type** : str
            Specifies the type of scaling applied.
            Supported options include:
                - ``"rms_forces"``
                - ``"std_forces"``
                - ``"rms_forces_by_element"``
                - ``"std_forces_by_element"``

            - **shift_type** : str
            Specifies the type of shifting applied.
            Supported options include:
                - ``"mean_delta_energy_per_atom"``
                - ``"mean_delta_energy_per_atom_by_element"``

            - **scale_trainable** : bool
            If ``True``, the scaling factor is trainable.

            - **shift_trainable** : bool
            If ``True``, the shifting factor is trainable.

            - **scale_dict** : str or dict
            - If set to ``"auto"``, the scale values are computed automatically from the training dataset.
            - If set to a dictionary, you must manually provide scale values for **all** elements,
                where keys are atomic numbers (int) and values are floats.
                It is recommended to configure these values directly in the `statistic.yaml` file.
                Usually, keeping this set to ``"auto"`` is sufficient and preferred.

            - **shift_dict** : str or dict
            - If set to ``"auto"``, the shift values are computed automatically from the training dataset.
            - If set to a dictionary, you must manually provide shift values for **all** elements,
                where keys are atomic numbers (int) and values are floats.
                It is recommended to configure these values directly in the `statistic.yaml` file.
                Usually, keeping this set to ``"auto"`` is sufficient and preferred.


    long_range: dict
        Latent Ewald Summation (LES) configuration.

        LES is a method to incorporate long-range electrostatics in MLIPs.
        For detailed parameters, please refer to the
        `LES GitHub repository <https://github.com/ChengUCB/les>`_.

        If fine-grained control is not needed, set this parameter to ``null``
        to use default values.

        For example: ::
            long_range:
                les: # for an introduction to the arguments, see the official repo: https://github.com/ChengUCB/les
                enable_les: false
                les_arguments: null # If you want to use LES but are not sure how to set the arguments, just set this to null to use the default values
                # les_arguments:
                #   n_layers: 3
                #   n_hidden': [32, 16]
                #   add_linear_nn: true
                #   output_scaling_factor: 0.1
                #   sigma: 1.0
                #   dl: 2.0
                #   remove_mean: true
                #   epsilon_factor: 1.0
                #   use_atomwise: false
                #   compute_bec: false
                #   bec_output_index: null


        Dictionary keys:

            - **enable_les** : bool
            Whether to enable LES. Default is ``False``.

            - **les_arguments** : dict or None
            Dictionary of LES-specific arguments.
            Use ``null`` to apply default values.

    universal_embedding : dict
        Embedding configuration for universal features.

        For example: ::

            universal_embedding:
                invariant: 
                level:
                    type: discrete # for discrete, label must use 0, 1, 2 in atoms.info ......
                    per: graph
                    in_dim: 1
                    out_dim: 64
                    num_classes: 5

                charges:
                    type: continuous 
                    per: atom
                    in_dim: 1
                    out_dim: 64
                    bias: false
                    act: 'silu'

                total_charge:
                    type: continuous
                    per: graph
                    in_dim: 1
                    out_dim: 64
                    bias: false
                    act: 'silu'


                equivariant:
                magmoms_1:
                    per: atom
                    rank: 1
                    element_trainable: true
                    channel_trainable: true

                electric_field:
                    per: graph
                    rank: 1
                    element_trainable: true
                    channel_trainable: true

                magnetic_field:
                    per: graph
                    rank: 1 # allow 1 or 2
                    element_trainable: true
                    channel_trainable: true

        Each element is a dictionary with keys:

            - **property** : str
            Property name. Examples:
                - ``"charges"`` (per-atom charge input)
                - ``"total_charge"`` (graph-level total charge)
                - ``"electronic_state"`` (graph-level discrete electronic state, label should start from 0 to num_classes-1)

            - **type** : str
            Type of property. Options:
                - ``"continuous"``
                - ``"discrete"``

            - **per** : str
            Specifies the level of property:
                - ``"atom"`` → per-atom
                - ``"graph"`` → per-graph

            - **in_dim** : int
            Input dimension size.
            Although arbitrary dimensions are allowed, the input features
            should be **rotationally invariant**.
            For example, vibrational states may be represented by three integer labels.

            - **out_dim** : int
            Output embedding dimension size.
            Typically this is set equal to ``num_channel``.

            - **bias** : bool (only for continuous property)
            Whether to include bias term. Default is ``False``.

            - **act** : str (only for continuous property)
            Activation function to apply. Example: ``"silu"``.

            - **num_classes** : int (only for discrete property)
            Number of classes for discrete property.

    conservations : dict
        Conservation-related constraints applied to physical quantities.

        For each physical quantity, **only one of the enable options can be set
        to ``true``**.

        - The first option corresponds to the method described in the paper,
        which enforces stronger physical constraints.

        - The second option redistributes the redundant predicted part
        uniformly over all atoms.

        Dictionary keys:

            - **charges** : dict
            Charge conservation options.

                - **enable_Qeq** : bool
                Enable the charge equilibration method (Qeq).
                This is the stricter physically motivated scheme.
                Default is ``False``.

                - **enable_Quni** : bool
                Constrain excess charge to be uniformly distributed across atoms.
                Default is ``False``.

    atomic_numbers: list
        It is usually read automatically from the dataset, but if you want to retain more elements in advance, 
        you can specify them manually. Remember to provide the isolated atomic energies as well.

    mixed_precision: dict
        Training with mixed data from multiple levels of calculation.

        For example: ::
            # single fidelity
            mixed_precision:
                level_names: [PBE]
                atomic_energies: 
                - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
                enable_multi_head: false
                enable_multi_fidelity: false

            # Multi-fidelity
            mixed_precision:
                level_names: [PBE, r2SCAN]
                atomic_energies: 
                - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
                - {2:0.0009,3:-0.3000,4:-0.04211,5:-0.2997,6:-1.3129,}
                enable_multi_head: false
                enable_multi_fidelity: true 

            # Multi-head
            mixed_precision:
                level_names: [PBE, r2SCAN]
                atomic_energies: 
                - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
                - {2:0.0009,3:-0.3000,4:-0.04211,5:-0.2997,6:-1.3129,}
                enable_multi_head: true
                enable_multi_fidelity: false

            # Multi-fidelity + Multi-head
            mixed_precision:
                level_names: [PBE, r2SCAN]
                atomic_energies: 
                - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
                - {2:0.0009,3:-0.3000,4:-0.04211,5:-0.2997,6:-1.3129,}
                enable_multi_head: true
                enable_multi_fidelity: true
    """

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
            
        # === Direct Polarizability ReadOut ===
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
            E_0 = scatter_sum(source=e0_node_energy, index=batch, dim=-1, dim_size=num_graphs)
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

            E_N = scatter_sum(source=node_energy, index=batch, dim=-1, dim_size=num_graphs)
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
            D_N = scatter_sum(source=node_dipole, index=batch, dim=0, dim_size=num_graphs)
            D = D_0 + D_N

        # === Polarizability ReadOut ===
        P = None
        if 'direct_polarizability' in self.target_property:
            p0n_node_polarizability = []
            p2n_node_polarizability = []
            for ii, (polarizability_readout0, polarizability_readout2) in enumerate(
                zip(self.polarizability_readout0s, self.polarizability_readout2s)
            ):
                if self.use_only_last_layer:
                    ii = -1
                p0n_node_polarizability.append(
                    polarizability_readout0(
                        descriptors[ii][0],
                    )[num_atoms_arange, node_level]
                )
                p2n_node_polarizability.append(
                    polarizability_readout2(
                        descriptors[ii][2],
                    )[num_atoms_arange, node_level, :, :]
                )
            p0_node_polarizability = torch.sum(torch.stack(p0n_node_polarizability, dim=-1), dim=-1)
            p2_node_polarizability = torch.sum(torch.stack(p2n_node_polarizability, dim=-1), dim=-1)
            P0_N = scatter_sum(
                source=p0_node_polarizability, index=batch, dim=0, dim_size=num_graphs
            )
            P2_N = scatter_sum(source=p2_node_polarizability, index=batch, dim=0, dim_size=num_graphs)
            I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
            P_N = P2_N + (P0_N / 3.0).view(-1, 1, 1) * I
            P = P_N
            if self.manual_symmetrize:
                P = P + P.permute(0, 2, 1).contiguous()  # only for reducible

        # === Direct Forces ReadOut ===
        D_F = None
        if 'direct_forces' in self.target_property:
            dfn = []
            for ii, direct_forces_readout in enumerate(self.direct_forces_readouts):
                if self.use_only_last_layer:
                    ii = -1
                if self.enable_nonlinearty_for_tensor:
                    dfn.append(
                        direct_forces_readout(
                            descriptors[ii][1],
                            descriptors[ii][0],
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :]
                    )
                else:
                    dfn.append(
                        direct_forces_readout(
                            descriptors[ii][1],
                            None,
                            None,
                            node_level,
                        )[num_atoms_arange, node_level, :]
                    )
            D_F = torch.sum(torch.stack(dfn, dim=-1), dim=-1)
  
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
            D_V2_N = scatter_sum(source=d_v2_node, index=batch, dim=0, dim_size=num_graphs)
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
            "direct_polarizability": P,
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

