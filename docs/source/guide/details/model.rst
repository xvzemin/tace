model
=====

.. note::
    You can view detailed descriptions of some parameters here. We also provide detailed explanations in the example 
    input files, code, so you can use these resources together for a better understanding.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   models/models.rst
   
Example
-------

.. code-block:: yaml
    
    config:
    _target_: tace.models.tace.TACE # only TACE are avaliable now

    atomic_numbers: null

    mixed_precision:
        # single fidelity
        level_names: [PBE]
        atomic_energies: # null for auto caculate
            - {1: -13.587222780835477, 6: -1029.4889999855063, 7: -1484.9814568572233, 8: -2041.9816003861047}
        enable_multi_head: false
        enable_multi_fidelity: false

    max_r_1: 2 # Truncation for node, 0-2 is recommended
    max_r_2: 3 # Truncation for edge, 3 is always recommended, 2 for higher speed
    num_layers: 2 # 2 is always recommended
    bias: true # use different systems with caution, false is more safe, but true may give better results
    num_channel: 64 # 64's accuracy is enough to achieve SOAT, recommended value is 64, for small number of parameters, use 48
    num_channel_hidden: ${model.config.num_channel} # should be <= num_channel, equal is the best

    radial_basis:
        radial_basis: j0   # [j0, jn, normalized_bessel]
        num_radial_basis: 8 # 8-10 is recommended
        # distance_transform: Agnesi # [Agnesi, Soft], this plug-in from mace, if you do not know what it means, never use
        distance_transform: null
        polynomial_cutoff: 5
        order: 0 # specify order for jn, for j0, it is negligible, for jn, you should use float64
        trainable: false
        apply_cutoff: true # whether use cutoff in radil basis or contraction, true is in radial basis

    angular_basis:
        type: ictd # always set this to ictd 
        norm: true # always set this to true

    radial_mlp:
        hidden: 
            - [64, 64, 64, 1024] # [64, 64, 64] for efficiency,
            - [64, 64, 64, 1024] #  [64, 64, 64m 1024] for accuracy, not affect spped
        act: 'silu'
        bias: false
        enable_layer_norm: false # if inter.add_source_target_embedding = True else False

    inter:
        max_paths: 1 # >= 1 (int) or null, 1 is recommended
        restriction: [null, null] # restriction for each layer, can be ``null`` or ``<=`` (r_1 <= r_2 for combinations)
        allow_nosym: true # allow path for each combination to product non-symmetric tensor
        kernel: einsum  # or broadcast, may optimize in future, einsum is always recommended
        residual: false # false is recommended
        add_source_target_embedding: false # cat extra source_target embedding to edge_feats
        normalizer: 
        type: fixed # fixaed or dynamic
        hidden: [64]
        act_1: 'silu'
        act_2: 'tanh'
        bias: false
        scale_shift_trainable: true # only useful for ``dynamic``

    prod:
        restriction:
        r_1_r_2: null  # `null` or `<=`
        r_o_r_1: null  # `null` or ``<=` or `==`
        kernel: einsum  # or broadcast, may optimize in future, einsum is always recommended
        allow_nosym: ${model.config.inter.allow_nosym}
        correlation: 3 # body-order, 3 is recommended
        element: true # always set this to true
        coupled: true # always set this to true

    ictd:
        weight: max  # [all, max, null]  # always set this to max

    readout_mlp:
        hidden: [16] # 16 is enough
        act: silu # silu is recommended
        bias: false # false is recommended, only useful for rank-0 tensor
        
    scale_shift:
        scale_type: rms_forces # add ``by_element`` can scale for each element
        shift_type: mean_delta_energy_per_atom # add ``by_element`` can shift for each element,  for energy only, set ``null`` or ``std_energy``
        scale_trainable: false
        shift_trainable: false
        scale_dict: auto
        shift_dict: auto

    short_range:
        enable_zbl: false # whether enable near-nuclear repulsion, for practical application, true is recommended

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

        # universal_embedding:
        #   invariant: 
        #     level:
        #       type: discrete # for discrete, label must use 0, 1, 2 in atoms.info ......
        #       per: graph
        #       in_dim: 1
        #       out_dim: 64
        #       num_classes: 5

        #     charges:
        #       type: continuous 
        #       per: atom
        #       in_dim: 1
        #       out_dim: 64
        #       bias: false
        #       act: 'silu'

        #     total_charge:
        #       type: continuous
        #       per: graph
        #       in_dim: 1
        #       out_dim: 64
        #       bias: false
        #       act: 'silu'


        #   equivariant:
        #     magmoms_1:
        #       per: atom
        #       rank: 1
        #       element_trainable: true
        #       channel_trainable: true

        #     electric_field:
        #       per: graph
        #       rank: 1
        #       element_trainable: true
        #       channel_trainable: true

        #     magnetic_field:
        #       per: graph
        #       rank: 1 # allow 1 or 2
        #       element_trainable: true
        #       channel_trainable: true

    conservations: # only one of enable_* can be true for each property
        charges:
        enable_Qeq: false # Qeq method, more physicscal
        enable_Quni: false # Uniform redistribution method


        # # single fidelity
        # mixed_precision:
        #     level_names: [PBE]
        #     atomic_energies: 
        #     - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
        #     enable_multi_head: false
        #     enable_multi_fidelity: false

        # # Multi-fidelity
        # mixed_precision:
        #     level_names: [PBE, r2SCAN]
        #     atomic_energies: 
        #     - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
        #     - {2:0.0009,3:-0.3000,4:-0.04211,5:-0.2997,6:-1.3129,}
        #     enable_multi_head: false
        #     enable_multi_fidelity: true 

        # # Multi-head
        # mixed_precision:
        #     level_names: [PBE, r2SCAN]
        #     atomic_energies: 
        #     - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
        #     - {2:0.0009,3:-0.3000,4:-0.04211,5:-0.2997,6:-1.3129,}
        #     enable_multi_head: true
        #     enable_multi_fidelity: false

        # # Multi-fidelity + Multi-head
        # mixed_precision:
        #     level_names: [PBE, r2SCAN]
        #     atomic_energies: 
        #     - {2:0.0008,3:-0.2973,4:-0.04130,5:-0.2911,6:-1.2775,}
        #     - {2:0.0009,3:-0.3000,4:-0.04211,5:-0.2997,6:-1.3129,}
        #     enable_multi_head: true
        #     enable_multi_fidelity: true