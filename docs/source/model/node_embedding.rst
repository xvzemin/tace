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

.. autoclass:: tace.models._e3nn.node.TensorNodeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.O2TensorNodeEmbedding
   :no-members:
   :show-inheritance:
