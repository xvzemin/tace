model
=====

This section describes the model architecture.

.. note::

   We recommend an overall architecture with either more than two layers and
   ``correlation = 2``, or two layers and ``correlation = 3``.

   - More than two layers with ``correlation = 2``
   - Two layers with ``correlation = 3``

The number of model parameters is mainly determined by the number of channels
and by whether the ResNet and product-basis modules are element-dependent.
When these modules are element-dependent, the number of model parameters can
increase substantially. This is the default and recommended setting, as it
does not affect computational speed. You can combine it with appropriate
nonlinearities to reduce the use of element-dependent modules.

Radial basis
------------

Select the spherical Bessel order through ``radial_basis.radial_basis``:
``j0``, ``j1``, ``j2``, and so on. ``j0`` retains the original radial basis
and normalization. Higher orders use a single-order spherical Bessel basis
normalized with the measure :math:`r^2\,dr`. ``num_radial_basis`` sets the
number of basis functions; no separate ``order`` configuration is needed.
``gaussian`` selects Gaussian basis functions.

Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: model
