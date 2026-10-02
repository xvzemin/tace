Node Embedding
==============

Among the node embedding methods provided below, we recommend using ``LinearNodeEmbedding`` in most cases. 
Other embedding strategies may not consistently achieve the best performance across different datasets. 
The linear embedding, which relies solely on element, is the most conservative and stable choice.

In principle, node embeddings can also incorporate information from the local environment (including scalar and tensor features). 
However, based on our current experiments, the linear approach remains the most reliable.

.. autoclass:: tace.models._e3nn.node.LinearNodeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.LinearSpinNodeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.NonLinearSpinNodeEmbedding
   :no-members:
   :show-inheritance:

Tensor embeddings aggregate element scalars multiplied by radial weights and
component-normalized spherical harmonics. The following four types share the
same output layout:

.. list-table:: Tensor embeddings
   :header-rows: 1

   * - ``node_embedding.type``
     - Angular construction
     - Radial MLP input
   * - ``spherical_tensor``
     - Spherical harmonics
     - Radial basis
   * - ``spherical_tensor_element2``
     - Spherical harmonics
     - Radial basis and both element embeddings
   * - ``wigner_tensor``
     - Inverse Wigner rotation of local scalars
     - Radial basis
   * - ``wigner_tensor_element2``
     - Inverse Wigner rotation of local scalars
     - Radial basis and both element embeddings

Element-independent weights do not remove element information from the node
features. Corresponding spherical and Wigner variants share learned weights
and produce the same output when supplied with the same normalized directions.
The names ``tensor`` and ``o2_tensor`` are not accepted.

.. autoclass:: tace.models._e3nn.node.SphericalTensorNodeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.Element2SphericalTensorNodeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.WignerTensorNodeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.Element2WignerTensorNodeEmbedding
   :no-members:
   :show-inheritance:
