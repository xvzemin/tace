"""Multi-scale radial weights for tensor-product convolutions."""

from typing import NamedTuple

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..ops import config
from ..ops.common.edge_pairs import project_rows
from ..ops.common.radial import (
    materialize,
    radial_layers,
    radial_linear_weight,
    radial_parts,
    radial_program,
)
from ..ops.common.weights import ConvolutionWeights
from ..ops.registry import resolve_backend


class _HiddenBranches(NamedTuple):
    layers: tuple
    widths: tuple
    affine: tuple

    def forward(self, value):
        *features, envelopes = value.split((*self.widths, len(self.widths)), dim=-1)
        blocks = []
        for i, (layers, hidden) in enumerate(zip(self.layers, features)):
            for layer in layers:
                hidden = layer(hidden)
            envelope = envelopes[:, i : i + 1]
            blocks.append(hidden * envelope)
            if self.affine[i]:
                blocks.append(envelope)
        return torch.cat(blocks, dim=-1)

    def to_edge_program(self, inputs, program, value):
        envelope_start = sum(self.widths)
        offset, blocks = 0, []
        for i, (layers, width) in enumerate(zip(self.layers, self.widths)):
            hidden = radial_program(
                layers, inputs, program, program.slice(value, offset, width)
            )
            envelope = program.slice(value, envelope_start + i, 1)
            blocks.append(
                program.binary(
                    "mul", hidden, program.gather(envelope, (0,) * program.size(hidden))
                )
            )
            if self.affine[i]:
                blocks.append(envelope)
            offset += width
        return program.concatenate(blocks)


class _PairBranches(NamedTuple):
    network: _HiddenBranches
    features: torch.Tensor
    inverse: torch.Tensor

    mode = "full"

    def forward(self, envelopes):
        features = self.features.index_select(0, self.inverse)
        return self.network.forward(torch.cat((features, envelopes), -1))

    def to_edge_program(self, inputs, program, envelopes):
        slot = len(inputs)
        inputs.extend((self.features, self.inverse))
        features = program.input(
            slot, ("indexed", slot + 1, slot), sum(self.network.widths)
        )
        return self.network.to_edge_program(
            inputs, program, program.concatenate((features, envelopes))
        )

    def narrow_edges(self, start, stop):
        return self._replace(inverse=self.inverse[start:stop])


class MultiCutoffRadial(torch.nn.Module):
    r"""Sum independent radial network outputs weighted by their edge envelopes.

    Parameters
    ----------
    networks : sequence of torch.nn.Module
        Scalar networks with the same output width. Each ends in an affine
        projection: a PyTorch linear layer, the linear output of an e3nn
        ``FullyConnectedNet``, or a layer exposing ``get_weight()`` in
        input-by-output layout. Hidden layers follow the convolution's radial
        network interface.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> model = MultiCutoffRadial([nn.Linear(4, 8), nn.Linear(6, 8)])
    >>> features = (torch.randn(5, 4), torch.randn(5, 6))
    >>> envelopes = torch.rand(5, 2)
    >>> model(features, envelopes).shape
    torch.Size([5, 8])
    >>> radial, projection, network = model.convolution_weights(features, envelopes)
    >>> torch.testing.assert_close(radial @ projection, model(features, envelopes))
    """

    def __init__(self, networks):
        super().__init__()
        self.networks = torch.nn.ModuleList(networks)
        if not self.networks:
            raise ValueError("At least one radial network is required.")
        outputs = []
        for network in self.networks:
            layers = radial_layers(network)
            last = layers[-1] if layers else None
            if not (
                isinstance(last, torch.nn.Linear)
                or hasattr(last, "get_weight")
                or (hasattr(last, "h_in") and last.act is None)
            ):
                raise ValueError(
                    "Each radial network must end in an affine projection."
                )
            weight = radial_linear_weight(last)
            outputs.append(weight.size(1))
        if len(set(outputs)) != 1:
            raise ValueError("Radial networks must have the same output width.")

    @staticmethod
    def relative_envelopes(envelopes):
        """Factor the outer cutoff out of nested envelopes of shape (edges, scales)."""
        outer = envelopes[:, -1:]
        inner = envelopes[:, :-1]
        # Zero inner branches have no support near the outer cutoff. Avoid
        # an intermediate reciprocal there, including in force derivatives.
        denominator = torch.where(inner != 0, outer, 1.0)
        return torch.cat((inner / denominator, torch.ones_like(outer)), dim=1)

    def _prepare_inputs(self, features, envelopes, edge_index, edge_pairs):
        if (
            len(features) != len(self.networks)
            or envelopes.ndim != 2
            or envelopes.shape[1] != len(features)
        ):
            raise ValueError(
                "Provide one feature matrix and one envelope column per network."
            )
        rows = (
            envelopes.shape[0]
            if edge_pairs is None
            else edge_pairs.representatives.numel()
        )
        for feature in features:
            if isinstance(feature, torch.Tensor) and (
                feature.ndim != 2
                or (feature.shape[0] != rows and feature.shape[0] != 1)
            ):
                raise ValueError(
                    "Supply one radial feature row per edge or pair, or one shared row."
                )
        if edge_pairs is None:
            return features
        if edge_pairs.inverse.shape != (envelopes.size(0),):
            raise ValueError("Supply one envelope row per directed edge.")
        if any(isinstance(x, tuple) for x in features):
            if edge_index is None:
                raise ValueError("Partitioned radial features require edge_index.")
            representatives = edge_pairs.representatives
            pair_index = (
                edge_index[0][representatives],
                edge_index[1][representatives],
            )
            features = tuple(materialize(x, pair_index) for x in features)
        return tuple(x.expand(rows, -1) for x in features)

    def forward(self, features, envelopes, *, edge_index=None, edge_pairs=None):
        """Evaluate dense path weights of shape ``(edges, weights)``.

        Parameters
        ----------
        features : sequence
            One feature matrix or sequence of indexed radial partitions per branch.
        envelopes : torch.Tensor
            Cutoff values of shape ``(edges, scales)``.
        edge_index : torch.Tensor, optional
            Source and target indices for indexed radial partitions.
        edge_pairs : EdgePairs, optional
            Map features on representative edges to directed envelope rows.
        """
        features = self._prepare_inputs(features, envelopes, edge_index, edge_pairs)
        result = None
        for i, (network, feature) in enumerate(zip(self.networks, features)):
            if isinstance(feature, tuple) and isinstance(feature[0][1], str):
                if edge_index is None:
                    raise ValueError("Partitioned radial features require edge_index.")
                feature = materialize(feature, edge_index)
            term = network(feature)
            if edge_pairs is not None:
                term = edge_pairs.expand(term)
            term = term * envelopes[:, i : i + 1]
            result = term if result is None else result + term
        return result

    def convolution_weights(
        self,
        features,
        envelopes,
        *,
        radial_mode="last",
        edge_index=None,
        edge_pairs=None,
        backend=None,
    ):
        """Prepare operands for a convolution with one shared angular contraction.

        Parameters
        ----------
        features : sequence
            One feature matrix per network, or indexed radial partitions
            with storage ``edge``, ``source``, ``target`` or ``shared``.
        envelopes : torch.Tensor
            Cutoff values of shape ``(edges, scales)``, in branch order.
        radial_mode : {"last", "full"}
            Fuse the stacked final projection, or also defer all hidden layers.
        edge_index : torch.Tensor, optional
            Source and target indices for partitioned inputs.
        edge_pairs : EdgePairs, optional
            Map features on representative edges to directed envelope rows.
        backend : {"torch", "cuda"}, optional
            Backend for evaluating shared pair features in ``full`` mode.
        """
        features = self._prepare_inputs(features, envelopes, edge_index, edge_pairs)
        if radial_mode not in ("last", "full"):
            raise ValueError("radial_mode must be last or full.")
        blocks, matrices, hidden, widths, affine = [], [], [], [], []
        partitions = edge_pairs is None and any(
            isinstance(x, tuple) or x.shape[0] != envelopes.shape[0] for x in features
        )
        for i, (network, x) in enumerate(zip(self.networks, features)):
            *layers, last = radial_layers(network)
            matrix = radial_linear_weight(last)
            matrices.append(matrix)
            bias = getattr(last, "bias", None) is not None
            if bias:
                matrices.append(last.bias.unsqueeze(0))
            if radial_mode == "last":
                if isinstance(x, tuple):
                    if edge_index is None:
                        raise ValueError(
                            "Partitioned radial features require edge_index in last mode."
                        )
                    x = materialize(x, edge_index)
                for layer in layers:
                    x = layer(x)
                if edge_pairs is not None:
                    x = edge_pairs.expand(x)
                envelope = envelopes[:, i : i + 1]
                blocks.append(x * envelope)
                if bias:
                    blocks.append(envelope)
            else:
                if (
                    edge_pairs is not None
                    and layers
                    and statically_known_true(
                        x.size(0) * matrix.size(0) * x.element_size() * len(features)
                        <= config.RADIAL_PROJECT_MAX_BYTES
                    )
                ):
                    x = project_rows(
                        x,
                        None,
                        layers,
                        None,
                        resolve_backend(backend, x.device),
                        "full",
                        edge_pairs.representatives,
                    )
                    layers = ()
                parts = radial_parts(x)
                widths.append(sum(t.shape[1] for t, _ in parts))
                hidden.append(tuple(layers))
                affine.append(bias)
                if partitions:
                    blocks.extend(parts)
                else:
                    blocks.append(x)
        provider = None
        if radial_mode == "full":
            provider = _HiddenBranches(tuple(hidden), tuple(widths), tuple(affine))
            if edge_pairs is not None:
                provider = _PairBranches(
                    provider, torch.cat(blocks, -1), edge_pairs.inverse
                )
                return ConvolutionWeights(envelopes, torch.cat(matrices, 0), provider)
            blocks.append((envelopes, "edge") if partitions else envelopes)
        radial = (
            tuple(blocks)
            if partitions and radial_mode == "full"
            else torch.cat(blocks, -1)
        )
        return ConvolutionWeights(radial, torch.cat(matrices, 0), provider)
