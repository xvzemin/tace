# cartTACE

The Cartesian version of TACE is currently being refactored for better future compatibility.

Select `model.config._target_: tace.models.cartTACE` with the ordinary TACE
configuration. Node features, convolutions and ACE products use EQX Cartesian
operators with the same paths, parameters and normalization as the spherical
model. `TACE_USE_EQX=1` enables fused convolution and ACE evaluation.

Local O(2) and SO(2) interactions are reserved interfaces, not implementations.
