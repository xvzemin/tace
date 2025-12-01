dataset
=======

Example
-------

.. code-block:: yaml

  dataset:
    storage_mode: lmdb # memory or lmdb, if your dataset is large (> 100w), the recommended approach is to use lmdb, as this avoids repeatedly constructing the graph.  
    shard_dirs: # if lmdb model, specify a list of path where you save you graph
      - graphCache
    # If using LMDB mode, you must allocate a reasonable pre-storage size.  
    # Generally, if the number of neighboring atoms is around 60, about 200 KB per graph is sufficient.
    shard_size: 1000 # number of graphs stored per file in LMDB mode, should be large, here is just an example, if small, it will be slow and may be error
    cache_size: 1024 # cache for faster load data from dataloader
    avg_graph_size_in_KB: 200 # in KB, the total disk usage equals the size of this multiplied by the total number of your structures.
    lmdb_wait_timeout: 86400 # in seconds; when training with multi-GPU, the maximum waiting time.
    type: ase # ase or ase-db
    split_seed: ${misc.global_seed} # this random seed is useful if auto split
    train_file: dataset/train_300K.xyz
    valid_file: null
    test_files: 
      - dataset/test_300K.xyz
      - dataset/test_600K.xyz
      - dataset/test_1200K.xyz
      - dataset/test_dih.xyz
    valid_ratio: 0.1 # auto split from train file if valid file is null, The priority order is:  no_valid_set > valid_file > valid_from_index > ``valid_ratio.
    valid_from_index: false # split train and val from train.index and valid.index in current directory
    no_valid_set: false # The prerequisite for enabling this is that you are using a learning rate scheduler that does not depend on the validation set.
    keys: # all default key name is the property name
      energy_key: energy
      forces_key: forces
      direct_forces_key: forces
      stress_key: stress
      direct_stress_key: direct_stress
      virials_key: virials
      direct_virials_key: direct_virials
      conservative_dipole_key: conservative_dipole
      direct_dipole_key: ditect_dipole
      polarization_key: polarization
      conservative_polarizability_key: conservative_polarizability
      direct_polarizability_key: direct_polarizability
      electric_field_key: electric_field
      born_effective_charges_key: born_effective_charges
      charges_key: charges
      total_charge_key: total_charge
      spin_multiplicity_key: spin_multiplicity
      magnetic_field_key: magnetic_field
      magmoms_0_key: magmoms_0
      magmoms_1_key: magmoms_1
      total_magmom_0_key: total_magmom_0
      total_magmom_1_key: total_magmom_1
      magnetic_forces_0_key: magnetic_forces_0
      magnetic_forces_1_key: magnetic_forces_1
      level_key: level

 
.. note::
   - The priority order is:  ``no_valid_set`` > ``valid_file`` > ``valid_from_index`` > ``valid_ratio``.