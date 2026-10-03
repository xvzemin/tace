"""Exact, inference-only execution of frozen TACE-OAM models."""

import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .streaming import CompressedMessage

logger = logging.getLogger(__name__)


def spatial_order(positions):
    """Order atoms by interleaved spatial bits to keep receiver halos compact."""
    lower = positions.min(axis=0)
    extent = np.maximum(positions.max(axis=0) - lower, 1e-12)
    points = ((positions - lower) / extent * ((1 << 21) - 1)).astype(np.uint64)

    def spread(value):
        value = (value | value << 32) & np.uint64(0x1F00000000FFFF)
        value = (value | value << 16) & np.uint64(0x1F0000FF0000FF)
        value = (value | value << 8) & np.uint64(0x100F00F00F00F00F)
        value = (value | value << 4) & np.uint64(0x10C30C30C30C30C3)
        return (value | value << 2) & np.uint64(0x1249249249249249)

    keys = spread(points[:, 0]) | spread(points[:, 1]) << 1 | spread(points[:, 2]) << 2
    return np.argsort(keys, kind="stable")


class LinearFunction(torch.autograd.Function):
    """Frozen linear map whose input adjoint requires no saved activations."""

    @staticmethod
    def forward(ctx, module, features):
        ctx.module = module
        return module.apply_matrix(features) + module.bias

    @staticmethod
    def backward(ctx, gradient):
        return None, ctx.module.transpose(gradient)


class RadialFunction(torch.autograd.Function):
    """Reuse a frozen radial value and its derivative with respect to length."""

    @staticmethod
    def forward(ctx, length, value, derivative):
        ctx.save_for_backward(derivative)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        (derivative,) = ctx.saved_tensors
        return (gradient * derivative).sum(-1, keepdim=True), None, None


class Linear(torch.nn.Module):
    """Pack an equivariant linear map into one matrix per irrep."""

    def __init__(self, reference):
        super().__init__()
        self.irreps_in, self.irreps_out = reference.irreps_in, reference.irreps_out
        weight = reference.weight
        if reference.use_matrix_weight:
            weight = torch.cat([w.flatten() for w in weight])
        self.groups = []
        offset = 0
        instructions = []
        for ins in reference.linear.instructions:
            size = math.prod(ins.path_shape)
            instructions.append(
                (ins, weight[offset : offset + size].view(ins.path_shape))
            )
            offset += size
        for ir in dict.fromkeys(ir for _, ir in self.irreps_out):
            inputs = [
                (k, mul, s)
                for k, ((mul, r), s) in enumerate(
                    zip(self.irreps_in, self.irreps_in.slices())
                )
                if r == ir
            ]
            outputs = [
                (k, mul, s)
                for k, ((mul, r), s) in enumerate(
                    zip(self.irreps_out, self.irreps_out.slices())
                )
                if r == ir
            ]
            matrix = weight.new_zeros(
                (sum(m for _, m, _ in inputs), sum(m for _, m, _ in outputs))
            )
            for ins, value in instructions:
                ii = next(
                    (i for i, (k, _, _) in enumerate(inputs) if k == ins.i_in), None
                )
                jj = next(
                    (j for j, (k, _, _) in enumerate(outputs) if k == ins.i_out), None
                )
                if ii is not None and jj is not None:
                    a, b = (
                        sum(m for _, m, _ in inputs[:ii]),
                        sum(m for _, m, _ in outputs[:jj]),
                    )
                    matrix[a : a + value.shape[0], b : b + value.shape[1]] += (
                        value * ins.path_weight
                    )
            self.register_buffer(f"weight_{len(self.groups)}", matrix.detach())
            self.groups.append((ir.dim, inputs, outputs))
        with torch.no_grad():
            bias = reference(weight.new_zeros(1, self.irreps_in.dim)).flatten()
        self.register_buffer("bias", bias)

    def apply_matrix(self, x, transpose=False):
        irreps = self.irreps_in if transpose else self.irreps_out
        result = [None] * len(irreps)
        for k, (dim, inputs, outputs) in enumerate(self.groups):
            weight = getattr(self, f"weight_{k}")
            if transpose:
                inputs, outputs, weight = outputs, inputs, weight.T
            if inputs:
                parts = [x[:, s].reshape(x.shape[0], mul, dim) for _, mul, s in inputs]
                value = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
                value = value.transpose(1, 2).reshape(x.shape[0] * dim, weight.shape[0])
                value = (
                    (value @ weight)
                    .reshape(x.shape[0], dim, weight.shape[1])
                    .transpose(1, 2)
                )
                for (j, _, _), part in zip(
                    outputs, value.split([mul for _, mul, _ in outputs], 1)
                ):
                    result[j] = part.flatten(1)
            else:
                for j, mul, _ in outputs:
                    result[j] = x.new_zeros((x.shape[0], mul * dim))
        for k, (mul, ir) in enumerate(irreps):
            if result[k] is None:
                result[k] = x.new_zeros((x.shape[0], mul * ir.dim))
        return torch.cat(result, 1)

    def forward(self, x):
        return LinearFunction.apply(self, x)

    def transpose(self, x):
        """Apply the exact input adjoint without a dummy forward."""
        return self.apply_matrix(x, transpose=True)


class Radial(torch.nn.Module):
    """Fold fixed element embeddings into the first radial affine map."""

    def __init__(self, network, update, identity, *, projection=False):
        super().__init__()
        first = network[0]
        weight = first.get_weight().detach()
        with torch.no_grad():
            source = update.source_embedding(identity)
            target = update.target_embedding(identity)
        width = weight.shape[0] - source.shape[1] - target.shape[1]
        self.register_buffer("weight", weight[:width].contiguous())
        self.register_buffer(
            "bias",
            first.bias.detach()
            if first.bias is not None
            else weight.new_zeros(weight.shape[1]),
        )
        self.register_buffer("target", target @ weight[width : width + target.shape[1]])
        self.register_buffer("source", source @ weight[width + target.shape[1] :])
        self.tail = torch.nn.Sequential(*list(network)[1 : -1 if projection else None])
        self.register_buffer(
            "projection", network[-1].get_weight().detach() if projection else None
        )
        if projection and network[-1].bias is not None:
            self.projection = torch.cat(
                (self.projection, network[-1].bias.detach()[None]), 0
            )
        self.append_one = projection and network[-1].bias is not None

    def forward(self, radial, source_type, target_type, index):
        value = torch.addmm(self.bias, radial, self.weight)
        value = value + self.target[target_type][index[1]]
        value = value + self.source[source_type][index[0]]
        value = self.tail(value)
        if self.append_one:
            value = torch.cat((value, torch.ones_like(value[:, :1])), -1)
        return value


class Storage:
    """Node matrices on the device, host, or a private memory-mapped workspace."""

    def __init__(self, shape, dtype, device, directory=None, zero=False):
        self.mapping = None
        if directory is not None:
            fd, name = tempfile.mkstemp(suffix=".bin", dir=directory)
            os.close(fd)
            self.path = Path(name)
            self.mapping = np.memmap(
                name,
                mode="w+",
                shape=shape,
                dtype=np.float64 if dtype == torch.float64 else np.float32,
            )
            self.tensor = torch.from_numpy(self.mapping)
        else:
            self.path = None
            self.tensor = torch.empty(shape, dtype=dtype, device=device)
        if zero:
            self.tensor.zero_()

    def read(self, index, device):
        if isinstance(index, torch.Tensor):
            index = index.to(self.tensor.device, dtype=torch.long)
        return self.tensor[index].to(device)

    def write(self, begin, end, value):
        self.tensor[begin:end].copy_(value)

    def add(self, index, value):
        if isinstance(index, slice):
            self.tensor[index].add_(value.to(self.tensor.device))
        else:
            self.tensor.index_add_(
                0,
                index.to(self.tensor.device, dtype=torch.long),
                value.to(self.tensor.device),
            )

    def close(self):
        del self.tensor
        if self.mapping is not None:
            self.mapping.flush()
            del self.mapping
            self.path.unlink()


class Edge(torch.nn.Module):
    """Stream the folded radial networks, convolution and density reduction."""

    def __init__(self, convolution, radial, density, apply_cutoff):
        super().__init__()
        from eqx.conv.network import radial_program
        from eqx.conv.program import Program

        program = Program()
        values = [None] * 8

        def parameter(tensor):
            slot = len(values)
            values.append(tensor.detach().reshape(1, -1).contiguous())
            return program.input(slot, "shared", tensor.numel())

        features = program.input(0, "source", convolution.input_dim)
        inputs = program.input(1, "edge", radial.weight.shape[0])
        cutoff = program.input(2, "edge", 1)
        vectors = program.input(3, "edge", 3)

        def network(module, source_slot):
            width, hidden = module.weight.shape
            value = program.matmul(inputs, parameter(module.weight), 1, width, hidden)
            value = program.binary("add", value, parameter(module.bias))
            value = program.binary(
                "add", value, program.input(source_slot, "source", hidden)
            )
            value = program.binary(
                "add", value, program.input(source_slot + 1, "target", hidden)
            )
            value = radial_program(module.tail, values, program, value)
            if module.append_one:
                value = program.concatenate((value, program.constant(1, 1)))
            return value

        weights = network(radial, 4)
        amplitude = program.gather(cutoff, (0,) * convolution.amplitude_dim)
        output = program.constant(convolution.output_dim, 0)
        message = program.add(
            "convolution",
            convolution.output_dim,
            (
                features,
                weights,
                parameter(radial.projection),
                amplitude,
                output,
                vectors,
            ),
            (
                "o3",
                convolution.harmonic_metadata,
                4,
                False,
                tuple(radial.projection.shape),
                0,
            ),
        )
        value = network(density, 6)
        value = program.unary("tanh", program.binary("mul", value, value))
        if apply_cutoff:
            value = program.binary("mul", value, cutoff)
        self.metadata = repr(
            (tuple(program.nodes), ((message, 0, "target"), (value, 1, "target")))
        )
        self.constants = len(values) - 8
        for slot, tensor in enumerate(values[8:]):
            self.register_buffer(f"constant_{slot}", tensor)
        self.radial, self.density = radial, density

    def inputs(self, features, radial, cutoff, vectors, source_type, target_type):
        rows = max(len(source_type), len(target_type))

        def pad(value):
            return (
                torch.nn.functional.pad(value, (0, 0, 0, rows - value.shape[0]))
                if value.shape[0] != rows
                else value
            )

        inputs = [
            pad(features),
            radial,
            cutoff,
            vectors,
            pad(self.radial.source[source_type]),
            pad(self.radial.target[target_type]),
            pad(self.density.source[source_type]),
            pad(self.density.target[target_type]),
        ]
        inputs.extend(
            getattr(self, f"constant_{slot}") for slot in range(self.constants)
        )
        return inputs, rows

    def forward(
        self, features, radial, cutoff, vectors, index, source_type, target_type
    ):
        from eqx.conv.edge import evaluate

        inputs, rows = self.inputs(
            features, radial, cutoff, vectors, source_type, target_type
        )
        message, density = evaluate(self.metadata, inputs, index[0], index[1], rows)
        return message[: len(target_type)], density[: len(target_type)]


class Graph:
    """Receiver tiles with exact periodic neighbor selection.

    Large cells use a periodic spatial tree; small cells retain every image
    returned by the ordinary neighbor list. Geometry is generated per tile.
    """

    def __init__(self, atoms, cutoff, tile_size, dtype, device):
        from scipy.spatial import cKDTree

        self.atoms = atoms
        self.cutoff, self.tile_size = cutoff, tile_size
        self.num_nodes = len(atoms)
        self.device = device
        self.dtype = dtype
        # Subtract positions before casting: large cells must not lose bond
        # precision merely because their absolute coordinates are large.
        self.positions = torch.as_tensor(
            atoms.positions, dtype=torch.float64, device=device
        )
        self.cell = torch.as_tensor(
            np.asarray(atoms.cell), dtype=torch.float64, device=device
        )
        self.tiles = {}
        self.device_tiles = {}
        self.cache_device = False
        self.tree = None
        self.fractional = None
        self.periodic = bool(atoms.pbc.all())
        if self.periodic:
            inverse = np.linalg.inv(atoms.cell)
            heights = 1 / np.linalg.norm(inverse, axis=0)
            if np.min(heights) > 2 * cutoff:
                self.fractional = atoms.positions @ inverse
                diagonal = np.allclose(
                    np.asarray(atoms.cell), np.diag(np.diag(atoms.cell)), atol=0, rtol=0
                ) and np.all(np.diag(atoms.cell) > 0)
                self.points = self.fractional % 1
                if diagonal:
                    self.points = self.points * np.diag(atoms.cell)
                    self.tree = cKDTree(self.points, boxsize=np.diag(atoms.cell))
                    self.radius = cutoff
                else:
                    self.tree = cKDTree(self.points, boxsize=1.0)
                    self.radius = (
                        cutoff / np.linalg.svd(atoms.cell, compute_uv=False).min()
                    )
        elif not atoms.pbc.any():
            self.points = atoms.positions
            self.tree = cKDTree(self.points)
            self.radius = cutoff
        if self.tree is None:
            from matscipy.neighbours import neighbour_list

            source, target, shifts = neighbour_list("ijS", atoms, cutoff)
            order = np.argsort(target, kind="stable")
            self.source = source[order].astype(np.int32)
            self.target = target[order].astype(np.int32)
            self.shifts = shifts[order].astype(np.int32)
            self.offsets = np.searchsorted(self.target, np.arange(self.num_nodes + 1))

    def tile(self, begin, stop=None):
        end = min(begin + self.tile_size, self.num_nodes if stop is None else stop)
        key = (begin, end)
        if key in self.device_tiles:
            return self.device_tiles[key]
        if key not in self.tiles:
            if self.tree is not None:
                neighbors = self.tree.query_ball_point(
                    self.points[begin:end], self.radius, workers=1, return_sorted=True
                )
                counts = np.fromiter((len(row) for row in neighbors), dtype=np.int64)
                target = np.repeat(np.arange(begin, end), counts)
                source = (
                    np.concatenate(neighbors).astype(np.int64)
                    if counts.sum()
                    else np.empty(0, dtype=np.int64)
                )
                shifts = (
                    -np.rint(self.fractional[target] - self.fractional[source]).astype(
                        np.int32
                    )
                    if self.fractional is not None
                    else np.zeros((len(source), 3), dtype=np.int32)
                )
                vectors = (
                    self.atoms.positions[target]
                    - self.atoms.positions[source]
                    + shifts @ self.atoms.cell
                )
                keep = (source != target) & (
                    np.einsum("ij,ij->i", vectors, vectors) < self.cutoff**2
                )
                source, target, shifts = source[keep], target[keep], shifts[keep]
            else:
                start, stop = self.offsets[begin], self.offsets[end]
                source, target, shifts = (
                    self.source[start:stop],
                    self.target[start:stop],
                    self.shifts[start:stop],
                )
            global_source = torch.as_tensor(
                source, device=self.device, dtype=torch.long
            )
            nodes, inverse = global_source.unique(return_inverse=True)
            index = torch.stack(
                (inverse, torch.as_tensor(target - begin, device=self.device))
            )
            self.tiles[key] = (
                nodes.cpu().int(),
                index.cpu().int(),
                torch.from_numpy(np.ascontiguousarray(shifts)),
            )
        nodes_cpu, index_cpu, shifts_cpu = self.tiles[key]
        nodes = nodes_cpu.to(self.device, dtype=torch.long)
        index = index_cpu.to(self.device, dtype=torch.long)
        vectors = (
            self.positions[begin:end][index[1]]
            - self.positions[nodes][index[0]]
            + shifts_cpu.to(self.device, dtype=self.positions.dtype) @ self.cell
        ).to(self.dtype)
        result = end, nodes_cpu, nodes, index, vectors
        if self.cache_device:
            self.device_tiles[key] = result
        return result


class OAM:
    """Evaluate frozen OAM checkpoints with bounded GPU workspaces.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded TACE tensor model with EQX O3 convolution and ACE enabled.
        The supplied model is repacked in place and is no longer trainable.
    tile_size : int, optional
        Maximum number of receiver nodes per tile.
    storage : {"auto", "cuda", "cpu", "disk"}, optional
        Storage of layer-boundary node states and adjoints. Auto selects CUDA
        when the estimated state fits, otherwise host memory.
    workspace : str or pathlib.Path, optional
        Parent directory for private temporary files in disk mode.
    cache_messages : bool, optional
        Retain compressed node messages to avoid convolution replay.
    domain_size : int, optional
        Approximate number of owned atoms in independent domains. Zero disables
        decomposition. Every domain includes the complete model receptive field.
    """

    def __init__(
        self,
        model,
        *,
        tile_size=4096,
        storage="auto",
        workspace=None,
        cache_messages=True,
        domain_size=0,
    ):
        from tace.models._e3nn.inter import O3CgtpInteraction
        from tace.models.linear import e3nnElementLinear, e3nnLinear

        if (
            storage not in ("auto", "cuda", "cpu", "disk")
            or tile_size <= 0
            or domain_size < 0
        ):
            raise ValueError("Invalid storage, tile_size, or domain_size.")
        self.core = model.readout_fn.eval().requires_grad_(False)
        self.rep = self.core.representation
        if (
            self.core.use_alllayer
            or self.rep.use_magnetic_interaction
            or len(self.core.energy_readouts) != 1
            or hasattr(self.core, "les")
            or hasattr(self.core, "one_body_magmoms_readout")
        ):
            raise ValueError(
                "This evaluator requires a nonmagnetic last-layer energy readout."
            )
        if len(self.rep.interactions) < 2 or any(
            type(inter) is not O3CgtpInteraction or not inter.use_eqx
            for inter in self.rep.interactions
        ):
            raise ValueError("Enable EQX O3 convolution before packing the model.")
        if any(
            not hasattr(inter, "edge_density") or inter.rejector.eqx_tp.normalize
            for inter in self.rep.interactions
        ):
            raise ValueError(
                "This evaluator requires density normalization and pre-normalized directions."
            )
        if any(
            not hasattr(prod, "eqx_ace") or prod.correlation != 2
            for prod in self.rep.products
        ):
            raise ValueError("This evaluator requires EQX correlation-two ACE.")
        if (
            not hasattr(self.rep.node_embedding, "elem_emb1")
            or getattr(self.rep, "uee_embeddings", None) is not None
            or hasattr(self.rep, "uie_embedding")
            or self.rep.radial_basis.use_distance_transform
            or self.rep.radial_basis.apply_cutoff
        ):
            raise ValueError(
                "This evaluator requires the unconditioned OAM embedding and radial basis."
            )
        coefficients = {
            id(coef)
            for prod in self.rep.products
            for name in ("coefs", "shared_coefs")
            for coef in getattr(prod, name, ())
        }
        for parent in list(self.core.modules()):
            for name, child in list(parent.named_children()):
                if type(child) is e3nnLinear and id(child) not in coefficients:
                    setattr(parent, name, Linear(child))
                elif type(child) is e3nnLinear and child.use_matrix_weight:
                    weight = torch.cat([w.flatten() for w in child.weight])
                    del child.weight
                    child.weight = torch.nn.Parameter(weight, requires_grad=False)
                    child.use_matrix_weight = False
                elif type(child) is e3nnElementLinear and child.use_matrix_weight:
                    weight = torch.cat(
                        [w.reshape(child.num_elements, -1) for w in child.weight], -1
                    )
                    del child.weight
                    child.weight = torch.nn.Parameter(weight, requires_grad=False)
                    if child.bias is not None:
                        child.bias = torch.nn.Parameter(
                            child.bias.reshape(child.num_elements, -1),
                            requires_grad=False,
                        )
                    child.use_matrix_weight = False
        parameter = next(self.core.parameters())
        self.dtype, self.device = parameter.dtype, parameter.device
        if self.device.type != "cuda":
            raise ValueError("The OAM inference evaluator requires CUDA.")
        self.tile_size, self.storage, self.workspace = tile_size, storage, workspace
        self.cache_messages = cache_messages
        self.domain_size = domain_size
        self.cache_basis = True
        self.atomic_numbers = self.core.atomic_numbers
        self.cutoff = model.get_cutoff()
        self.identity = torch.eye(
            len(self.atomic_numbers), dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            self.embedding = self.rep.node_embedding.elem_emb1(self.identity)
        self.radial, self.density = [], []
        for inter, update in zip(self.rep.interactions, self.rep.edge_updates):
            self.radial.append(
                Radial(inter.edge_info.mlp, update, self.identity, projection=True)
            )
            self.density.append(Radial(inter.edge_density.mlp, update, self.identity))
        self.edges = [
            Edge(inter.rejector.eqx_tp, radial, density, inter.apply_density_cutoff)
            for inter, radial, density in zip(
                self.rep.interactions, self.radial, self.density
            )
        ]

    @classmethod
    def from_checkpoint(cls, path, *, device="cuda", dtype="float32", **kwargs):
        """Load a checkpoint before creating an inference-only execution plan."""
        from tace.lightning.lit_model import load_tace
        from tace.utils.env import enable_acceleration

        enable_acceleration(enable_eqx=True, force=True)
        model = load_tace(
            path,
            device=device,
            dtype=dtype,
            target_property=["energy", "forces", "stress", "virials"],
        )
        return cls(model, **kwargs)

    def layer(
        self,
        layer,
        source,
        previous,
        vectors,
        index,
        source_type,
        target_type,
        saved=None,
        basis=None,
    ):
        inter, prod = self.rep.interactions[layer], self.rep.products[layer]
        attrs = self.identity[target_type]
        length = vectors.square().sum(-1, keepdim=True).sqrt() + 1e-9
        if basis is None:
            radial, cutoff = self.rep.radial_basis(
                length, attrs, index, self.atomic_numbers
            )
            radial = self.rep.edge_embedding(None, attrs, radial, index, cutoff)
        else:
            embedded = RadialFunction.apply(length, *basis)
            radial, cutoff = embedded[:, :-1], embedded[:, -1:]
        rejector = inter.rejector
        if hasattr(inter, "norm1"):
            source = inter.reshape1.inverse(inter.norm1(inter.reshape1(source)))
        source = rejector.reshape_in(inter.linear_up(source))
        if saved is None:
            message, density = self.edges[layer](
                source,
                radial,
                cutoff,
                vectors / length,
                index,
                source_type,
                target_type,
            )
            message = inter.linear_down(rejector.reshape_out(message))
        else:
            weights = self.radial[layer](radial, source_type, target_type, index)
            message = CompressedMessage.apply(
                inter,
                index,
                saved,
                source,
                weights,
                self.radial[layer].projection,
                cutoff.expand(-1, rejector.eqx_tp.amplitude_dim),
                vectors / length,
            )
            density = torch.tanh(
                self.density[layer](radial, source_type, target_type, index).square()
            )
            if inter.apply_density_cutoff:
                density = density * cutoff
            density = density.new_zeros((len(target_type), 1)).index_add(
                0, index[1], density
            )
        compressed = message
        density = density * inter.beta + inter.alpha
        density = density.masked_fill(density == 0, 1e-9)
        message = inter.linear_nonlinearity(
            inter.nonlinearity(inter._normalize_messages(message, density))
        )

        def residual(module, features):
            if inter.resnet_linear_type == "aware":
                return module(features, attrs, target_type)
            return module(features)

        if hasattr(inter, "resnetBA"):
            message = message + residual(inter.resnetBA, previous)
        skip = None
        if hasattr(inter, "resnetBB"):
            skip = residual(inter.resnetBB, previous)
        elif hasattr(inter, "resnetAB"):
            skip = residual(inter.resnetAB, message)
        if hasattr(inter, "norm2"):
            message = inter.reshape2.inverse(inter.norm2(inter.reshape2(message)))
        output = prod(
            message,
            attrs,
            skip,
            torch.zeros_like(target_type),
            node_type=target_type,
        )
        if layer == len(self.rep.interactions) - 1 and hasattr(self.rep, "final_norm"):
            output = self.rep.final_reshape.inverse(
                self.rep.final_norm(self.rep.final_reshape(output))
            )
        return output, compressed

    def readout(
        self, features, vectors, index, source_type, target_type, fidelity, isolated
    ):
        core = self.core
        energy = core.energy_readouts[0](
            features, torch.full_like(target_type, fidelity)
        )[:, fidelity]
        zbl = 0
        if hasattr(core, "zbl"):
            length = vectors.square().sum(-1, keepdim=True).sqrt() + 1e-9
            types = torch.cat((target_type, source_type))
            zindex = torch.stack((index[0] + len(target_type), index[1]))
            zbl = core.zbl(
                length,
                self.identity[types],
                zindex,
                self.atomic_numbers,
                node_type=types,
            )[: len(target_type)]
            if core.scale_zbl:
                energy = energy + zbl
        if hasattr(core, "scale_shift"):
            scale = core.scale_shift
            if scale.has_scale:
                energy = energy * (
                    0
                    if isolated and not scale.all_atoms
                    else scale.scale[fidelity, target_type]
                )
            if scale.has_shift and not (isolated and not scale.all_atoms):
                energy = energy + scale.shift[fidelity, target_type]
        if hasattr(core, "zbl") and not core.scale_zbl:
            energy = energy + zbl
        return energy + core.atomic_energy_layer.atomic_energy[fidelity, target_type]

    @torch.no_grad()
    def __call__(self, atoms, *, fidelity=0):
        """Return energy, forces, stress, and virials for one ASE structure."""
        if self.domain_size and len(atoms) > self.domain_size:
            return self.domains(atoms, fidelity)
        if len(atoms) <= self.tile_size:
            return self.evaluate(atoms, fidelity=fidelity)
        order = spatial_order(atoms.positions)
        result = self.evaluate(atoms[order], fidelity=fidelity)
        result["forces"] = result["forces"][np.argsort(order)]
        return result

    @torch.no_grad()
    def evaluate(self, atoms, *, fidelity=0, counts=None):
        """Execute shrinking receiver sets, with a complete halo at each layer."""
        started = time.perf_counter()
        n = len(atoms)
        if not n:
            raise ValueError("At least one atom is required.")
        lookup = {int(z): i for i, z in enumerate(self.atomic_numbers.cpu().tolist())}
        types = torch.tensor(
            [lookup[int(z)] for z in atoms.numbers],
            dtype=torch.long,
            device=self.device,
        )
        if not 0 <= fidelity < self.core.atomic_energy_layer.atomic_energy.shape[0]:
            raise ValueError("The fidelity index is outside the checkpoint heads.")
        graph = Graph(atoms, self.cutoff, self.tile_size, self.dtype, self.device)
        if counts is None:
            counts = [n] * len(self.rep.interactions)
        dims = [prod.irreps_out.dim for prod in self.rep.products[:-1]]
        bytes_per = torch.empty((), dtype=self.dtype).element_size()
        sizes = [dim * count for dim, count in zip(dims, counts)]
        estimate = bytes_per * (sum(sizes) + 2 * max(sizes))
        if self.cache_messages:
            estimate += bytes_per * sum(
                inter.linear_down.irreps_out.dim * count
                for inter, count in zip(self.rep.interactions[:-1], counts)
            )
        storage = self.storage
        if storage == "auto":
            free, _ = torch.cuda.mem_get_info(self.device)
            storage = "cuda" if estimate < max(0, free - (4 << 30)) * 0.7 else "cpu"
        store_device = self.device if storage == "cuda" else torch.device("cpu")
        graph.cache_device = storage == "cuda" and n <= 65536
        if storage == "disk" and self.workspace is None:
            raise ValueError("Disk storage requires a workspace directory.")
        if storage == "cpu":
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            if estimate > available * 0.8:
                raise MemoryError(
                    "Layer states exceed the host memory budget; use domain_size or disk storage."
                )
        if (
            storage == "disk"
            and estimate > shutil.disk_usage(self.workspace).free * 0.9
        ):
            raise MemoryError(
                "The workspace does not have enough free disk space for layer states."
            )
        directory = (
            tempfile.TemporaryDirectory(dir=self.workspace, prefix="eqx-oam-")
            if storage == "disk"
            else None
        )
        allocated = []

        def allocate(dim, rows, zero=False):
            value = Storage(
                (rows, dim),
                self.dtype,
                store_device,
                directory.name if directory else None,
                zero,
            )
            allocated.append(value)
            return value

        def release(value):
            if value is not None:
                value.close()
                allocated.remove(value)

        def load(state, index):
            if state is None:
                return self.embedding[
                    types[
                        index.to(self.device)
                        if isinstance(index, torch.Tensor)
                        else index
                    ]
                ]
            return state.read(index, self.device)

        forces = torch.zeros((n, 3), dtype=self.dtype, device=store_device)
        virial = torch.zeros((3, 3), dtype=torch.float64, device=self.device)
        energy = torch.zeros((), dtype=torch.float64, device=self.device)
        states, messages = [None], []
        embeddings = {}

        def basis(begin, end, vectors, index):
            if not self.cache_basis or storage != "cuda" or n > 16384:
                return None
            key = begin, end
            if key not in embeddings:
                attrs = self.identity[types[begin:end]]

                def embed(length):
                    value, cutoff = self.rep.radial_basis(
                        length, attrs, index, self.atomic_numbers
                    )
                    value = self.rep.edge_embedding(None, attrs, value, index, cutoff)
                    return torch.cat((value, cutoff), -1)

                length = vectors.detach().norm(dim=-1, keepdim=True) + 1e-9
                embeddings[key] = torch.autograd.functional.jvp(
                    embed, length, torch.ones_like(length)
                )
            return embeddings[key]

        try:
            for layer, dim in enumerate(dims):
                current = allocate(dim, counts[layer])
                cache = (
                    allocate(
                        self.rep.interactions[layer].linear_down.irreps_out.dim,
                        counts[layer],
                    )
                    if self.cache_messages
                    else None
                )
                for begin in range(0, counts[layer], self.tile_size):
                    end, ids, nodes, index, vectors = graph.tile(begin, counts[layer])
                    value, compressed = self.layer(
                        layer,
                        load(states[-1], nodes if storage == "cuda" else ids),
                        load(states[-1], slice(begin, end)),
                        vectors,
                        index,
                        types[nodes],
                        types[begin:end],
                        basis=basis(begin, end, vectors, index),
                    )
                    current.write(begin, end, value)
                    if cache is not None:
                        cache.write(begin, end, compressed)
                states.append(current)
                messages.append(cache)
            adjoint = None
            for layer in reversed(range(len(self.rep.interactions))):
                gradient = (
                    allocate(dims[layer - 1], counts[layer - 1], zero=True)
                    if layer
                    else None
                )
                for begin in range(0, counts[layer], self.tile_size):
                    end, ids, nodes, index, vectors = graph.tile(begin, counts[layer])
                    node_ids = nodes if storage == "cuda" else ids
                    source = (
                        load(states[layer], node_ids).detach().requires_grad_(layer > 0)
                    )
                    previous = (
                        load(states[layer], slice(begin, end))
                        .detach()
                        .requires_grad_(layer > 0)
                    )
                    vectors.requires_grad_(True)
                    with torch.enable_grad():
                        saved = (
                            messages[layer].read(slice(begin, end), self.device)
                            if layer < len(messages) and messages[layer] is not None
                            else None
                        )
                        value, _ = self.layer(
                            layer,
                            source,
                            previous,
                            vectors,
                            index,
                            types[nodes],
                            types[begin:end],
                            saved,
                            basis(begin, end, vectors, index),
                        )
                        if layer == len(self.rep.interactions) - 1:
                            value = self.readout(
                                value,
                                vectors,
                                index,
                                types[nodes],
                                types[begin:end],
                                fidelity,
                                n == 1 and index.numel() == 0,
                            )
                            energy.add_(value.detach().double().sum())
                            seed = torch.ones_like(value)
                        else:
                            seed = adjoint.read(slice(begin, end), self.device)
                        inputs = [source, previous, vectors] if layer else [vectors]
                        grads = torch.autograd.grad(
                            value, inputs, seed, allow_unused=True
                        )
                    if layer:
                        dx, dh, dv = grads
                        if dx is not None:
                            gradient.add(node_ids, dx)
                        if dh is not None:
                            gradient.add(slice(begin, end), dh)
                    else:
                        dv = grads[0]
                    if dv is not None:
                        fs = dv.new_zeros((len(nodes), 3)).index_add_(0, index[0], dv)
                        ft = dv.new_zeros((end - begin, 3)).index_add_(0, index[1], -dv)
                        forces.index_add_(
                            0,
                            node_ids.to(store_device, dtype=torch.long),
                            fs.to(store_device),
                        )
                        forces[begin:end].add_(ft.to(store_device))
                        virial.sub_(vectors.detach().double().T @ dv.double())
                    del value, source, previous, vectors, grads, seed
                release(adjoint)
                release(states[layer])
                if layer < len(messages):
                    release(messages[layer])
                adjoint = gradient
            volume = abs(np.linalg.det(atoms.cell))
            stress = (
                -0.5 * (virial + virial.T) / volume
                if volume > 0
                else virial.new_zeros(3, 3)
            )
            torch.cuda.synchronize(self.device)
            self.statistics = dict(
                atoms=n,
                storage=storage,
                state_estimate_bytes=estimate,
                edges=sum(tile[1].shape[1] for tile in graph.tiles.values()),
                seconds=time.perf_counter() - started,
            )
            return dict(
                energy=energy.item(),
                forces=forces.cpu().numpy(),
                stress=stress.cpu().numpy(),
                virials=(0.5 * (virial + virial.T)).cpu().numpy(),
            )
        finally:
            for value in allocated:
                value.close()
            if directory is not None:
                directory.cleanup()

    @torch.no_grad()
    def domains(self, atoms, fidelity):
        """Sum owned energies and halo forces without graph-wide hidden states."""
        from ase import Atoms
        from scipy.spatial import cKDTree

        started = time.perf_counter()
        cell = np.asarray(atoms.cell)
        if (
            not atoms.pbc.all()
            or not np.array_equal(cell, np.diag(np.diag(cell)))
            or np.any(np.diag(cell) <= 0)
        ):
            raise ValueError(
                "Domain execution currently requires an orthorhombic periodic cell."
            )
        lengths = np.diag(cell)
        depth = len(self.rep.interactions)
        divisions = np.maximum(
            1,
            np.rint(
                lengths / (np.prod(lengths) * self.domain_size / len(atoms)) ** (1 / 3)
            ).astype(int),
        )
        width = lengths / divisions
        halo = depth * self.cutoff
        if np.any(width / 2 + halo >= lengths / 2):
            return self.evaluate(atoms, fidelity=fidelity)
        positions = atoms.positions % lengths
        bins = np.minimum((positions / width).astype(int), divisions - 1)
        labels = np.ravel_multi_index(bins.T, divisions)
        tree = cKDTree(positions, boxsize=lengths)
        forces = np.zeros((len(atoms), 3))
        virials = np.zeros((3, 3))
        energy, largest, evaluations = 0.0, 0, 0
        unique = np.unique(labels)
        for domain, label in enumerate(unique):
            center = (np.asarray(np.unravel_index(label, divisions)) + 0.5) * width
            ids = np.asarray(
                tree.query_ball_point(center, np.max(width / 2 + halo), p=np.inf),
                dtype=np.int64,
            )
            relative = positions[ids] - center
            relative -= np.rint(relative / lengths) * lengths
            distance = np.maximum(np.abs(relative) - width / 2, 0).max(axis=1)
            keep = distance <= halo + 1e-10
            ids, relative, distance = ids[keep], relative[keep], distance[keep]
            owned = labels[ids] == label
            shells = np.where(
                owned, 0, np.maximum(1, np.ceil((distance - 1e-10) / self.cutoff))
            ).astype(int)
            order = spatial_order(relative)
            order = order[np.argsort(shells[order], kind="stable")]
            ids, relative, distance = ids[order], relative[order], distance[order]
            counts = [
                int(np.count_nonzero(shells <= depth - 1 - k)) for k in range(depth - 1)
            ] + [int(owned.sum())]
            local = Atoms(
                numbers=atoms.numbers[ids],
                positions=relative + center,
                cell=cell,
                pbc=False,
            )
            result = self.evaluate(local, fidelity=fidelity, counts=counts)
            energy += result["energy"]
            forces[ids] += result["forces"]
            virials += result["virials"]
            largest = max(largest, len(ids))
            evaluations += sum(counts)
            logger.info(
                "OAM domain %d/%d: %d owned, %d including halo, %.2f s",
                domain + 1,
                len(unique),
                counts[-1],
                len(ids),
                self.statistics["seconds"],
            )
            torch.cuda.empty_cache()
        self.statistics = dict(
            atoms=len(atoms),
            storage="domains",
            domains=int(np.unique(labels).size),
            largest_domain=largest,
            receiver_evaluations=evaluations,
            seconds=time.perf_counter() - started,
        )
        return dict(
            energy=energy,
            forces=forces,
            virials=virials,
            stress=-virials / abs(np.linalg.det(cell)),
        )


def ase_calculator(evaluator):
    """Return an ASE calculator backed by the frozen evaluator."""
    from ase.calculators.calculator import Calculator, all_changes
    from ase.stress import full_3x3_to_voigt_6_stress

    class CalculatorOAM(Calculator):
        implemented_properties = ["energy", "free_energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = evaluator(self.atoms)
            self.results["free_energy"] = self.results["energy"]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])

    return CalculatorOAM()
