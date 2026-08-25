model
=====

This section describes the model architecture. 

.. note::

   We recommend an overall architecture with either more than two layers and
   ``correlation = 2``, or two layers and ``correlation = 3``.

   - More than two layers with ``correlation = 2``
   - Two layers with ``correlation = 3``


The number of model parameters is mainly determined by the number of channels and 
by whether ``the ResNet and product basis modules are element-dependent``.
When these modules are element-dependent, the number of model parameters can increase substantially. 
This is the default and recommended setting, as it does not affect computational speed. 
You can combine it with appropriate nonlinearities to reduce the use of element-dependent modules.


Example
-------

.. code-block:: yaml

    model:
    # export TACE_USE_OEQ=1 to use openequivariance, 0 to disable
    # export TACE_USE_CUE=1 to use cuequivariance, 0 to disable
    # export TACE_USE_EQT=1 to use equitorch, 0 to disable for correlation > 2
    config:

        _target_: tace.models.e3nnTACE # [e3nnTACE] 
        wrapper:
        _target_: tace.models.TensorModel # [TensorModel]

        cutoff: 6.0 # (float), recommend 6.0
        max_neighbors: null # retained for model compatibility; neighbor filtering is disabled
        atomic_numbers: null # (null, list), null = read from dataset.
        
        fidelity:
        - {
            name: PBE, # (str) custom name
            atomic_energy: null # (null, Dict[int, float]), {1: -13.587222780835477, 6: -1029.4889999855063, 7: -1484.9814568572233, 8: -2041.9816003861047} 
        }

        node_embedding: 
        type: linear    # [linear, nonlinear, tensor, so2_tensor], linear is safer

        edge_embedding: 
        type: nonlinear # identity  # [identity, linear, nonlinear, element2], identity is safer

        edge_update:
        type: element2  # [identity, element2], identity is safer

        mmax: 3 # Truncation for SO2 irreps, 3 is recommended.
        Lmax: 2 # Truncation for node, 2 is recommended.
        lmax: 3 # Truncation for edge, 3 is recommended.
        num_layers: 3 # 2 for correlation >=3, can be higher when correlation=2
        num_channel: 48 # (int), 48~64, 64's accuracy is enough to achieve SOTA, never use channel > 64
        parity: false # (bool) in test, not for users, always false

        radial_basis:
        radial_basis: j0   # [j0, jn, gaussian]
        num_radial_basis: 8 # (int) 8-10 for j0 and jn, 8-64 for gaussian 
        distance_transform: null # [null, Agnesi, Soft], this plug-in from mace, if you do not know what it means, never use
        cutoff_fn: c2poly # [c2poly, c3poly, mollifier, cosine]
        polynomial_cutoff: 5 # (int)
        gaussian_width: 2.0
        order: 0 # (int), specify order for jn, for j0, it is negligible, for jn, <7 stable
        trainable: false # (bool)
        apply_cutoff: false # # (bool) true: cutoff was use in before radial_mlp, false: after radial_mlp
        hidden: [64, 64] # list(int)
        bias: true  # (bool)

        resnet: 
        type: BB # [BB, AB, BAB] A = atomic_basis, B = product_basis, control how resnet connect
        linear_type: aware # [aware, agnostic, identity]
        use_first_resnet: false

        layer_norm: # If training is unstable, prioritize specifying pre_norm_type. If it is still unstable, specify pre_norm_type.
        pre_norm_type: null     # [merge_rms_norm, merge_layer_norm]
        final_norm_type: null   # [merge_rms_norm, merge_layer_norm]
        use_first_pre_norm: false 
            
        atomic_basis:
        type: cgtp # [cgtp, so2] 
        # type: [cgtp, so2, cgtp, so2 ...]
        l1l2: null # [null, <=], restriction for each layer
        scatter_norm: avg_num_neighbors  # [sqrt_avg_num_neighbors, avg_num_neighbors, density]
        nonlinear: gate # [null, gate]
        edge_nonlinear: gate # edge scalar/tensor activations use scalar_act/tensor_act
        edge_info_type: mlp # [mlp, glu]
            
        # in test, not use
        node_wise_hidden: null
        edge_wise_hidden: null
        edge_ace_hidden: null
        num_head: null
        gate_m0: false # only controls the node gate; edge gates activate scalars directly
        scalar_act: null # unscaled name for 0e, or [0e, 0o]
        tensor_act: null # unscaled name for tensor-irrep 0e gates

        product_basis:
        type: cgtp  # [cgtp, gtp]
        # type: [cgtp, gtp, cgtp, gtp ...]
        l1l2: null  # [null, <=]
        correlation: 2 # (int) 3 is recommended for 2 layer, 2 is  is recommended for > 2 layer
        return_components: null # [null, list(int)], if not specify, last layer will only return target weight
        num_expert: null
        num_channel_per_expert: null
        resolution: null # list[int, int] only useful for GtpACE, 
        nonlinear: null
        use_shared_expert: false

        readout_emlp:
        bias: false
        hidden: [16]
        use_alllayer: false # (bool) recommended true, if true, evergy layer has readout, else, only last layer has
        use_uie: false # (bool) whether use uie readout
        
        scale_shift:
        enable: true
        scale_type: rms_forces # [null, rms_forces, rms_forcesrms_forces_by_element]
        shift_type: null # [null, mean_delta_energy_per_atom, forces, mean_delta_energy_per_atom_by_element]
        scale_trainable: false # (bool) 
        shift_trainable: false # (bool) 
        all_atoms: false
        scale_zbl: false

        short_range:
        zbl: # metal units
            enable: true    # recommended true
            trainable: false # recommended always false

        long_range:
        les: # for introduction to the arguments, see the official repo: https://github.com/ChengUCB/les
            enable: false
            les_arguments:
              use_dipole: false
              use_quad: false
              use_induced_charge: false
              use_induced_dipole: false
              use_anisotropic_polarizability: false
              sigma: 1.0
              dl: 2.0
              remove_self_interaction: true
              is_periodic: true # [null, true, false]
              N_max: 10
              use_epsilon_r_scaling: false

        universal_embedding: 
        charges:
            enable: false

        total_charge:
            enable: false

        spin_multiplicity:
            enable: false
            num_embeddings: 3 # (int) 

        electric_field:
            enable: false
            normalizer: 1.0 # (float)

        magnetic_field:
            enable: false
            normalizer: 1.0 # (float)

        initial_noncollinear_magmoms: # only in SO(3) group, not O(3) now
            enable: false
            normalizer: 1.0 # (float)

        special: # this field can be thoroughly ignored if you only interested in E/F/S training
        
        charges:
            method: lagrangian # [lagrangian, uniform_distribution]

        hessian:
            num_samples: 2 # traing E/F/Hessian, for each graph, <= num_samples atoms's hessian will be used to train


        # Unless you are traing foundation model, never use dropout
        # This is designed for direct pre-training
        dropout:  
        use_first_dropout: false
        stochastic_depth: 0.0
