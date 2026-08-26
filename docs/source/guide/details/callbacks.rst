callbacks
=========

A callback refers to a set of functions that are automatically invoked after the 
completion of each training epoch. At a minimum, you need at least one callback for saving the model.
If you have any special requirements, or if you want to use any built-in features provided by Lightning, 
you can add them here as callbacks.


Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: callbacks
