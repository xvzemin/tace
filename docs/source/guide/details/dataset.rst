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

The SOC architecture permits the broadest function space because it enforces
only coupled space--spin rotations. When it is fitted to non-SOC data,
``spin_rotation`` is required to sample the independent
:math:`SO(3)_{\mathrm{spin}}` symmetry. If the installed e3nn does not track
time-reversal parity, ``time_reversal`` is required as well. These
augmentations sample
:math:`O(3)_{\mathrm{space}}\times SO(3)_{\mathrm{spin}}
\times\mathbb Z_2^{\mathcal T}` in the training data; they do not make it an
exact architectural symmetry.

The explicit non-SOC architecture already enforces the full group
:math:`O(3)_{\mathrm{space}}\times SO(3)_{\mathrm{spin}}
\times\mathbb Z_2^{\mathcal T}` and therefore requires neither augmentation.
For zero-field SOC data, the complete time-reversal model requires no
augmentation, whereas the standard-e3nn SOC model uses ``time_reversal``.

No transformation acts on validation, test, or statistics loaders. Spin
rotations and reversals are global per structure; independent per-atom
transformations would change the physical exchange interactions.


.. note::
  
   - The priority order is:  ``no_valid_set`` > ``valid_file`` > ``valid_from_index`` > ``valid_ratio``.


Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: dataset
