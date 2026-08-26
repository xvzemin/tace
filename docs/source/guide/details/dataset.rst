dataset
=======

This field is used to store information about the sources of the train/valid/test set.
It also controls whether the constructed graphs are saved locally, whether they are 
loaded from specified files, and which keyscare used to read training labels from the input data.


.. note::
  
   - The priority order is:  ``no_valid_set`` > ``valid_file`` > ``valid_from_index`` > ``valid_ratio``.


Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: dataset
