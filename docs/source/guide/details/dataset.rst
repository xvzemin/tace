dataset
=======

This field is used to store information about the sources of the train/valid/test set.
It also controls whether the constructed graphs are saved locally, whether they are 
loaded from specified files, and which keyscare used to read training labels from the input data.

``augmentation`` is one symmetry-preserving transformation or a list of
transformations applied only when training samples are read. ``null`` and an
empty list disable augmentation. The available entries are:

``spin_rotation``
  Apply one uniformly sampled global spin rotation to
  ``initial_noncollinear_magmoms`` and ``noncollinear_magnetic_forces``.
  Positions are unchanged. This teaches the independent spin-rotation
  symmetry of data without spin--orbit coupling.

``time_reversal``
  Randomly reverse ``initial_noncollinear_magmoms`` and
  ``noncollinear_magnetic_forces`` together. This enforces global time
  reversal while preserving the consistency of the input and its derivative.

For a standard-e3nn :math:`O(3)` model, non-SOC data can use
``augmentation: [spin_rotation, time_reversal]`` and zero-field SOC data can
use ``augmentation: [time_reversal]``. An
:math:`O(3)\times\mathbb Z_2^{\mathcal T}` model already enforces time reversal,
so only ``spin_rotation`` is needed when it is fitted to non-SOC data. The
explicit non-SOC model requires neither augmentation. No transformation acts
on validation, test, or statistics loaders. Spin rotations and reversals are
global per structure; independent per-atom transformations would change the
physical exchange interactions.


.. note::
  
   - The priority order is:  ``no_valid_set`` > ``valid_file`` > ``valid_from_index`` > ``valid_ratio``.


Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: dataset
