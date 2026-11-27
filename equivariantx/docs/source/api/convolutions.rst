.. _equivariantx-api-convolutions:

Convolutions
============

``backend=None`` selects CUDA when available and optimized PyTorch otherwise.
Use ``backend="torch"`` to select PyTorch explicitly. Operands support
FP32 and FP64, with the caller's TF32 matrix policy. FP16, BF16 and reduced
precision autocast are unsupported. Geometry may remain FP64 with FP32 features.

For CGTP and UV-O(2), ``radial_mode="full"`` includes the radial network in
the convolution schedule. Use ``radial_mode="last"`` to keep its hidden
features and fuse only the output projection.
For O(3) CGTP, ``projection_workspace_bytes`` bounds the working space for
the tiled output projection, not the peak memory of the complete model.

``EQX_USE_CUDA_GRAPH=1`` enables native convolution replay after warmup.
Eligible local O(2) programs can reuse a capture across nearby node and edge
counts. Replay reserves additional memory.

Convolution weights
-------------------

A custom module may return weights directly: pass its output as ``radial``
and an empty ``(0, weight_numel)`` projection. For an indexed correction,
construct the row map once per graph. Different layers can reuse it with
different output widths:

.. code-block:: python

   rows = eqx.nn.indexed_rows(types[edge_index], table.size(0),
                                dtype=table.dtype, reduce="mean", backend=conv.backend)
   radial, projection, network = eqx.nn.indexed_weights(
       edge_features, weight, [(table, rows)], radial_network=hidden)
   output = conv(nodes, harmonics, radial, projection, edge_index,
                 radial_network=network)

For convolutions without a separate projection argument (such as UV-O(2)
or the matrix tensor product), use ``radial, network = weights.as_network()``
on the result of ``indexed_weights``.

Small maps can enter the final projection as additional columns. Larger
maps use integer lookups. For wide projections in training, a bounded row
map can also reduce repeated table gradients by matrix multiplication.

.. autofunction:: equivariantx.nn.indexed_weights

.. autofunction:: equivariantx.nn.indexed_rows

.. autofunction:: equivariantx.nn.indexed_radial

.. autoclass:: equivariantx.nn.MultiCutoffRadial
   :members: convolution_weights

.. autofunction:: equivariantx.utils.undirected_edge_pairs

For reciprocal weights, pass pair-sized radial inputs and ``edge_pairs=pairs``
to the convolution. If only the radial term is reciprocal, pass the map to
``indexed_weights`` instead. Its other terms remain directed:

.. code-block:: python

   pairs = undirected_edge_pairs(edge_index, edge_shifts, num_nodes=len(nodes))
   src, dst = atom_types[edge_index]
   terms = [(source_table, src), (target_table, dst)]
   r, p, net = indexed_weights(basis[pairs.representatives], projection, terms,
                               radial_network=hidden, edge_pairs=pairs)
   out = conv(nodes, harmonics, r, p, edge_index, radial_network=net)

For degree-specific radial bases, pass an
:class:`~equivariantx.nn.GroupedRadialMLP` as ``radial_network`` and an empty
projection of shape ``(0, conv.weight_numel)``. The convolution selects the
backend. ``radial_mode="last"`` retains hidden activations and ``"full"``
recomputes them during differentiation. If the pair-weight table exceeds the
workspace budget, the convolution evaluates weights in directed-edge tiles.
Both modes preserve the supplied path order.

Only share a deterministic, pointwise function whose inputs agree on edge
reversal. Reuse the map while the edge list and lattice translations are
unchanged. Angular features and the two directed messages are not identified.
Use directed inputs to differentiate with respect to independent edge lengths.

For a custom pointwise weight function, implement ``forward`` for reference
execution and optionally ``to_edge_program`` for streaming. The latter returns
an expression index. For example:

.. code-block:: python

   class QuadraticWeights(torch.nn.Module):
       def forward(self, x):
           return x + x.square()

       def to_edge_program(self, inputs, program, x):
           return program.binary("add", x, program.binary("mul", x, x))

See the convolution classes under :ref:`equivariantx-api-o3`,
:ref:`equivariantx-api-co3`, :ref:`equivariantx-api-o2`, and :ref:`equivariantx-api-co2`.
