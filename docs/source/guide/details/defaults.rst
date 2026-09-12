defaults
========

Use ``defaults`` to inherit a training configuration and specify only the
parameters that differ. TACE uses Hydra to compose the files.

.. note::

If you prefer, you can also manually merge all configuration options into
a single configuration file.

Inherit a configuration
~~~~~~~~~~~~~~~~~~~~~~~

Keep ``tace.yaml`` as the base configuration. An example in the same directory,
``tace-les.yaml``, enables LES without repeating the training settings:

.. literalinclude:: ../../../../example/train/tace-les.yaml
   :language: yaml

- ``tace`` selects ``tace.yaml``; omit the extension inside ``defaults``.
- ``@_global_`` merges its contents at the configuration root. For example,
  ``model.config`` remains ``model.config``, rather than being nested under
  another configuration group.
- ``_self_`` marks where the current file is merged. Put it last so the current
  settings override inherited values. Putting it first allows later files to
  override the current settings instead.

The base ``tace.yaml`` only needs ``_self_``. See Hydra's
`Defaults List <https://hydra.cc/docs/1.3/advanced/defaults_list/>`_ and
`package rules <https://hydra.cc/docs/1.3/advanced/overriding_packages/>`_
for the composition semantics.

Relative paths and chained configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the benchmark examples, the files are arranged as follows:

.. code-block:: text

   example/train/
     tace.yaml
     benchmark_configs/
       3bpa_cgtp.yaml
       3bpa_o2.yaml
       3bpa_o2_cgtp.yaml

``3bpa_cgtp.yaml`` loads the base from the parent directory:

.. yaml-config:: ../../../../example/train/benchmark_configs/3bpa_cgtp.yaml
   :path: defaults

The path is relative to the containing configuration; ``../`` selects its
parent directory. ``@_global_`` controls where the contents are merged, not
where the file is found.

A variant can then inherit ``3bpa_cgtp.yaml`` from the same directory.
For example, ``3bpa_o2_cgtp.yaml`` changes only the interaction type:

.. literalinclude:: ../../../../example/train/benchmark_configs/3bpa_o2_cgtp.yaml
   :language: yaml

The merge order is ``tace.yaml``, then ``3bpa_cgtp.yaml``, then
``3bpa_o2_cgtp.yaml``. The magnetic examples follow the same pattern:
``soc_o2_mtace.yaml`` supplies the shared magnetic settings, while
``nonsoc_o2_mtace.yaml`` and ``soc_w6j_mtace.yaml`` override their differences.

Keep the referenced files and their relative locations when copying an example.
These paths locate configuration files; they do not change dataset paths stored
inside those files.

.. note::

   Reusable examples use relative references instead of changing
   ``hydra.searchpath``. Hydra permits that setting only in the primary
   configuration, so putting it in a file that is later inherited causes a
   composition error. See `Config Search Path
   <https://hydra.cc/docs/1.3/advanced/search_path/>`_.

Merge and override rules
~~~~~~~~~~~~~~~~~~~~~~~~

- Dictionaries merge recursively: specifying one nested field keeps the other
  inherited fields.
- Later values override earlier values for the same key.
- Lists are replaced as a whole, not appended or merged element by element.
  When changing the target properties, update the corresponding loss names,
  weights, and keyword-argument lists together.
- ``null`` is an explicit value, not an instruction to inherit or delete a key.
  The parameter must support ``null``.

For example, add the following below ``defaults`` in a variant:

.. code-block:: yaml

   optimizer:
     lr: 5.0e-3

   model:
     config:
       radial_basis:
         hidden: [64, 64]

This changes the learning rate while retaining the optimizer class and its other
settings. It replaces the entire radial hidden-layer list. These are the
`OmegaConf merge rules
<https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#merging-configurations>`_.

An empty dictionary ``{}`` does not clear an inherited dictionary. The
energy/force-only 3BPA benchmark replaces the entire inherited validation-metric
mapping with a resolver instead:

.. yaml-config:: ../../../../example/train/benchmark_configs/3bpa_cgtp.yaml
   :path: synth_metric

Inspect the merged configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, select the benchmark directory and preview the
configuration:

.. code-block:: bash

   cd example/train/benchmark_configs
   tace-train -cn 3bpa_o2_cgtp.yaml --cfg job --resolve

This prints the merged configuration with interpolations resolved, without
starting training. Command-line values can override the merged settings:

.. code-block:: bash

   tace-train -cn 3bpa_o2_cgtp.yaml model.config.num_channel=32 --cfg job --resolve

Use ``--info defaults-tree`` to inspect the inheritance tree. Remove the
inspection flags when ready to train. For more override syntax, see the
`Hydra command-line guide
<https://hydra.cc/docs/1.3/advanced/override_grammar/basic/>`_.
