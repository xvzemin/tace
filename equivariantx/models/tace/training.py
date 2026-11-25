"""CUDA replay of energy, force and stress training steps."""

from contextlib import contextmanager

import torch

from equivariantx.ops.common.cuda.graph import retain_constants


def _bilinear_checkpoint(rep):
    from .foundation import representation_architecture

    return representation_architecture(rep) in {"7m", "rra"} and all(
        product.correlation == 2 and not hasattr(product, "stochastic_depth")
        for product in rep.products
    )


class FixedShapeTrainingStep:
    """Replay forward and parameter backward, then update live optimizer state.

    This single-GPU helper owns one fixed-shape input/gradient workspace. Use
    one instance for one model, optimizer and batch signature; it accepts
    changes to coordinates, topology, elements and labels of the same shape.
    The supplied objective takes (predictions, prepared_batch) and returns a
    scalar. With ``joint=True``, supply TACE's compile adapter and its
    supported loss module; the explicit joint-derivative program is run under
    capture instead. Warmup and capture do *not* update parameters. Only each explicit
    call advances the optimizer. The caller must not replace parameters, change
    training/precision policy, or use this concurrently with other gradient
    consumers. The returned metrics alias the workspace until the next call.
    No gradient accumulation, distributed collectives, AMP or dropout.
    """

    def __init__(
        self,
        model,
        optimizer,
        batch,
        objective,
        targets,
        *,
        joint=False,
        _workspace=None,
        _recordings=None,
    ):
        if not model.training or not torch.cuda.is_available():
            raise ValueError("A training-mode CUDA model is required")
        if torch.is_autocast_enabled("cuda") or (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise ValueError(
                "This single-GPU replay does not support AMP or distributed training"
            )
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        self.model, self.optimizer = model, optimizer
        params = tuple(p for p in model.parameters() if p.requires_grad)
        if not params or any(
            p.device != params[0].device or not p.is_cuda for p in params
        ):
            raise ValueError("All parameters must reside on one CUDA device")
        if set(params) != {
            p for group in optimizer.param_groups for p in group["params"]
        }:
            raise ValueError("The optimizer must own all trainable model parameters")
        self.params = params
        self.device = params[0].device
        batch = model.prepare_property_data(batch)
        self.static = {
            k: v.detach().clone() if torch.is_tensor(v) else v for k, v in batch.items()
        }
        self.graph = torch.cuda.CUDAGraph()
        self.constants = {}
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if _workspace is None
            else _workspace.capture_stream
        )
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        direct = (
            joint
            and type(optimizer) is torch.optim.AdamW
            and all(group.get("fused") for group in optimizer.param_groups)
            and not any(
                p._backward_hooks or p._post_accumulate_grad_hooks for p in params
            )
            and not any(p.requires_grad for p in objective.parameters())
        )

        def evaluate():
            # Grad buffers are allocated inside the graph pool by backward.
            # Replay therefore writes fresh gradients without accumulation.
            optimizer.zero_grad(set_to_none=True)
            if direct:
                from tace.models.compile.joint import joint_training

                outputs, loss, gradients = joint_training(
                    model, objective, self.static, return_gradients=True
                )
                # Preserve independent buffers if a compiled output is a view
                # of another gradient, a parameter, a prediction, or an input.
                owned = {
                    v.untyped_storage().data_ptr()
                    for v in (
                        *self.model.parameters(),
                        *self.model.buffers(),
                        *objective.buffers(),
                        *self.static.values(),
                        *outputs.values(),
                        loss,
                    )
                    if torch.is_tensor(v)
                }
                for parameter, gradient in gradients:
                    if gradient is not None:
                        ptr = gradient.untyped_storage().data_ptr()
                        if ptr in owned or not gradient.is_contiguous():
                            gradient = gradient.clone(
                                memory_format=torch.contiguous_format
                            )
                        owned.add(ptr)
                    parameter.grad = gradient
            elif joint:
                outputs, loss = model.forward_loss(self.static, objective)
                loss.backward(inputs=params)
            else:
                outputs = model(self.static)
                loss = objective(outputs, self.static)
                loss.backward(inputs=params)
            return {k: outputs[k].detach() for k in targets} | {"loss": loss.detach()}

        with (
            torch.cuda.device(self.device),
            torch.cuda.stream(self.stream),
            retain_constants(self.constants),
        ):
            evaluate()
            evaluate()
        self.stream.synchronize()
        warmed = dict(self.constants) if _workspace is not None else None

        def record():
            with (
                torch.cuda.device(self.device),
                retain_constants(self.constants),
                torch.cuda.graph(
                    self.graph,
                    stream=self.stream,
                    capture_error_mode="thread_local",
                    pool=None if _workspace is None else _workspace.pool,
                ),
            ):
                self.outputs = evaluate()
            if warmed is not None and (
                warmed.keys() != self.constants.keys()
                or any(t is not self.constants[k] for k, t in warmed.items())
            ):
                raise RuntimeError(
                    "A shared training graph created a persistent constant during capture"
                )
            self.gradients = tuple(p.grad for p in params)

        if _recordings is None:
            record()
        else:
            _recordings.append(record)

    def replay(self, batch):
        """Write predictions and fresh parameter gradients without an optimizer update."""
        if (
            not self.model.training
            or torch.is_autocast_enabled("cuda")
            or torch.backends.cuda.matmul.allow_tf32 != self.tf32
        ):
            raise ValueError("Training mode or numerical policy changed since capture")
        batch = self.model.prepare_property_data(batch)
        if batch.keys() != self.static.keys():
            raise ValueError("The training batch field set changed")
        with torch.no_grad():
            for k, dst in self.static.items():
                src = batch[k]
                if torch.is_tensor(dst):
                    if (
                        not torch.is_tensor(src)
                        or src.shape != dst.shape
                        or src.dtype != dst.dtype
                        or src.device != dst.device
                    ):
                        raise ValueError(f"Training input signature changed at {k}")
                    dst.copy_(src)
                elif src != dst:
                    raise ValueError(f"Static non-tensor input changed at {k}")
        for p, grad in zip(self.params, self.gradients):
            p.grad = grad
        self.graph.replay()
        return self.outputs

    def __call__(self, batch):
        outputs = self.replay(batch)
        self.optimizer.step()
        return outputs


def _pack(
    batch,
    atoms,
    edges,
    graphs,
    stress,
    joint=False,
    *,
    microbatches=1,
    total_atoms=None,
):
    """Append one disconnected ignored graph; no padding touches a real node."""
    n, e = len(batch["positions"]), batch["edge_index"].shape[1]
    if n < 1 or len(batch["ptr"]) != graphs + 1 or n > atoms or e > edges:
        raise ValueError(
            "Real minibatch exceeds this (graphs, atoms, edges) bucket"
        )
    # TACE represents absent labels with zero weights and a numeric placeholder.
    # Refuse those (and non-unit user weights) under this unweighted objective.
    for key in (
        "energy_weight",
        "forces_weight",
        *(("stress_weight",) if stress else ()),
        "entropy",
    ):
        if key in batch:
            torch._assert_async((batch[key] == 1).all(), f"Bucket requires unit {key}")
    extra = atoms + 1 - n  # At least one real index for the dummy self-image edges.
    dest = batch["edge_index"]
    # Even at the largest real edge count *every* padding atom must have a
    # nonzero neighborhood. The OAM-L radial/harmonic derivatives at empty
    # neighborhoods can be singular in force-loss double backward. Reserve
    # one image slot per possible dummy atom, independent of real topology.
    extra_edges = edges + atoms - e
    pad_nodes = (
        torch.arange(extra_edges, device=dest.device, dtype=dest.dtype).remainder(extra)
        + n
    )
    extra_shifts = batch["edge_shifts"].new_ones(extra_edges, 3)
    lattice = batch["lattice"]
    # Length sqrt(110) lies strictly outside the released 6-A cutoff, so many
    # coincident padding images cannot amplify a dummy message through layers.
    dummy_cell = torch.diag(lattice.new_tensor([5.0, 6.0, 7.0])).unsqueeze(0)

    def cat(key, tail):
        return torch.cat((batch[key], tail))

    result = dict(
        atomic_numbers=cat("atomic_numbers", batch["atomic_numbers"][:1].expand(extra)),
        positions=cat("positions", batch["positions"].new_zeros(extra, 3)),
        batch=cat("batch", batch["batch"].new_full((extra,), graphs)),
        ptr=cat("ptr", batch["ptr"].new_full((1,), atoms + 1)),
        edge_index=torch.cat((dest, pad_nodes.expand(2, -1)), dim=1),
        edge_shifts=cat("edge_shifts", extra_shifts),
        lattice=cat("lattice", dummy_cell),
        pbc=cat("pbc", batch["pbc"].new_ones(1, 3)),
        fidelity_idx=cat("fidelity_idx", batch["fidelity_idx"].new_zeros(1)),
        energy=cat("energy", torch.zeros_like(batch["energy"][:1])),
        forces=cat("forces", batch["forces"].new_zeros(extra, 3)),
    )
    if stress:
        result["stress"] = cat("stress", torch.zeros_like(batch["stress"][:1]))
    if joint:
        # Native NormalLoss averages E and F over the padded lengths while
        # S is normalized by the number of positive-weight real structures.
        # Undo only those two denominator changes, keeping the dummy at zero.
        real = lattice.new_ones(graphs + 1)
        real[-1] = 0
        result["energy_weight"] = real * ((graphs + 1) / graphs)
        result["forces_weight"] = real * ((atoms + 1) / n)
        result["entropy"] = torch.ones_like(real)
        if stress:
            result["stress_weight"] = real
    if microbatches > 1:
        if joint:
            result["energy_weight"].div_(microbatches)
            result["forces_weight"].mul_(n / total_atoms)
            if stress:
                result["stress_weight"] = real / microbatches
        else:
            result["macro_atoms"] = lattice.new_tensor(total_atoms)
    return result


def _efs_loss(pred, data, microbatches=1):
    counts = data["ptr"].diff()[:-1]
    energy = (
        pred["energy"].reshape(-1)[:-1] - data["energy"].reshape(-1)[:-1]
    ) / counts
    valid = data["batch"] != len(counts)
    force = (pred["forces"] - data["forces"]).square() * valid[:, None]
    energy = energy.square().mean()
    atoms = data["ptr"][-2]
    if microbatches > 1:
        energy = energy / microbatches
        atoms = data["macro_atoms"]
    loss = energy + force.sum() / (3 * atoms)
    if "stress" in data:
        stress = (pred["stress"][:-1] - data["stress"][:-1]).square().mean()
        loss = loss + (stress / microbatches if microbatches > 1 else stress)
    return loss


class BucketedEFSTrainingStep:
    """Train variable-size E/F(/S) minibatches in a CUDA Graph bucket.

    Uses an unweighted sum of energy-per-atom MSE, force-component MSE and
    optional stress-component MSE. Inputs must be fully labeled and all
    quantities use TACE's native units. This is deliberately *not* a generic
    replacement for arbitrary training losses or samplers. Set capacities to
    the largest real N/E; real graph count stays fixed. Bucket charges one
    additional disconnected padding graph, never contributes it to a loss or
    returns its predictions. Changing graph sizes, type IDs, edge indices,
    periodic shifts, geometry and targets are copied every call. All optimizer
    state lives on the original parameter objects and updates once per call.
    With ``microbatches > 1``, pass that many GPU batches per call, each with
    the example's structure count. Capacities apply to one microbatch. Energy
    and stress are averaged over all structures and forces over all atoms.
    """

    def __init__(
        self,
        model,
        optimizer,
        batch,
        *,
        max_atoms,
        max_edges,
        joint=False,
        microbatches=1,
        _workspace=None,
        _recordings=None,
    ):
        from .tace_oam_l.model import _OAMLRepresentation, _validate

        rep = model.readout_fn.representation
        if not isinstance(rep, _OAMLRepresentation) and not _bilinear_checkpoint(rep):
            _validate(rep)
        if model.get_cutoff() != 6.0:
            raise ValueError(
                "Bucket padding requires a released 6-A foundation model"
            )
        self.targets = set(model.get_target_property())
        if (
            self.targets not in ({"energy", "forces"}, {"energy", "forces", "stress"})
            or model.get_embedding_property()
        ):
            raise ValueError(
                "Bucket supports only fully labeled E/F or E/F/S batches"
            )
        if model.lmp or not model.training:
            raise ValueError("Bucket requires the non-LAMMPS training-mode adapter")
        if type(microbatches) is not int or microbatches < 1:
            raise ValueError("microbatches must be a positive integer")
        if microbatches > 1 and any(
            p._backward_hooks or p._post_accumulate_grad_hooks
            for p in model.parameters()
        ):
            raise ValueError(
                "Microbatch accumulation requires parameters without gradient hooks"
            )
        self.microbatches = microbatches
        self.cap = (
            max_atoms,
            max_edges,
            len(batch["ptr"]) - 1,
            "stress" in self.targets,
            joint,
        )
        objective = _efs_loss
        if microbatches > 1 and not joint:
            from functools import partial

            objective = partial(_efs_loss, microbatches=microbatches)
        if joint:
            from tace.models.compile.wrapper import CompileTensorModel
            from tace.utils.loss.normal import NormalLoss

            model = CompileTensorModel(model.readout_fn).train()
            targets = ["energy", "forces"] + (
                ["stress"] if "stress" in self.targets else []
            )
            names = ["mse_energy_per_atom", "mse_forces"] + (
                ["mse_stress"] if "stress" in self.targets else []
            )
            objective = NormalLoss(targets, names, [1.0] * len(targets)).train()
        self.step = FixedShapeTrainingStep(
            model,
            optimizer,
            _pack(
                batch,
                *self.cap,
                microbatches=microbatches,
                total_atoms=len(batch["positions"]) * microbatches,
            ),
            objective,
            sorted(self.targets),
            joint=joint,
            _workspace=_workspace,
            _recordings=_recordings,
        )
        if microbatches > 1:
            self.active = [
                (p, g)
                for p, g in zip(self.step.params, self.step.gradients, strict=True)
                if g is not None
            ]
            self.accumulated = [torch.empty_like(g) for _, g in self.active]

    @staticmethod
    def trim(outputs, batch):
        """Select predictions for real structures and atoms.

        >>> batch = {"positions": torch.zeros(2, 3)}
        >>> outputs = {"energy": torch.tensor([1.5, 0.0]), "forces": torch.ones(4, 3)}
        >>> real = BucketedEFSTrainingStep.trim(outputs, batch)
        >>> real["energy"].tolist(), tuple(real["forces"].shape)
        ([1.5], (2, 3))
        """
        return {
            k: v[: len(batch["positions"])]
            if k == "forces"
            else v[:-1]
            if k in {"energy", "stress"}
            else v
            for k, v in outputs.items()
        }

    def __call__(self, batch):
        if self.microbatches == 1:
            with torch.no_grad():
                packed = _pack(batch, *self.cap)
            return self.trim(self.step(packed), batch)
        if not isinstance(batch, (tuple, list)) or len(batch) != self.microbatches:
            raise ValueError(f"Expected {self.microbatches} structure microbatches")
        total_atoms = sum(len(b["positions"]) for b in batch)
        gradients = [g for _, g in self.active]
        outputs = []
        with torch.no_grad():
            for i, b in enumerate(batch):
                packed = _pack(
                    b,
                    *self.cap,
                    microbatches=self.microbatches,
                    total_atoms=total_atoms,
                )
                pred = self.trim(self.step.replay(packed), b)
                outputs.append({key: value.clone() for key, value in pred.items()})
                if i == 0:
                    torch._foreach_copy_(self.accumulated, gradients)
                else:
                    torch._foreach_add_(self.accumulated, gradients)
            for (parameter, _), grad in zip(self.active, self.accumulated, strict=True):
                parameter.grad = grad
        self.step.optimizer.step()
        return {
            key: sum(p[key] for p in outputs)
            if key == "loss"
            else torch.cat([p[key] for p in outputs])
            for key in outputs[0]
        }


class _SerialTrainingWorkspace:
    def __init__(self, model, optimizer, joint):
        from threading import get_ident

        from torch.nn.modules import module as module_hooks
        from torch.optim import optimizer as optimizer_hooks

        from .tace_oam_l.model import _OAMLRepresentation

        rep = getattr(getattr(model, "readout_fn", None), "representation", None)
        if (
            joint
            or type(optimizer) is not torch.optim.AdamW
            or not all(group.get("fused") for group in optimizer.param_groups)
            or not (isinstance(rep, _OAMLRepresentation) or _bilinear_checkpoint(rep))
        ):
            raise ValueError(
                "Shared workspace requires a supported foundation model with non-joint fused AdamW"
            )
        self.params = tuple(p for p in model.parameters() if p.requires_grad)
        if not self.params or not self.params[0].is_cuda:
            raise ValueError("Shared training workspaces require CUDA parameters")
        if (
            any(p._backward_hooks or p._post_accumulate_grad_hooks for p in self.params)
            or any(
                m._forward_hooks
                or m._forward_pre_hooks
                or m._backward_hooks
                or m._backward_pre_hooks
                for m in model.modules()
            )
            or optimizer._optimizer_step_pre_hooks
            or optimizer._optimizer_step_post_hooks
            or any(
                getattr(module_hooks, name, None)
                for name in (
                    "_global_forward_hooks",
                    "_global_forward_pre_hooks",
                    "_global_backward_hooks",
                    "_global_backward_pre_hooks",
                )
            )
            or optimizer_hooks._global_optimizer_pre_hooks
            or optimizer_hooks._global_optimizer_post_hooks
        ):
            raise ValueError("Shared workspace does not support training hooks")
        device = self.params[0].device
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=device)
        self.caller_stream = torch.cuda.current_stream(device)
        self.thread = get_ident()
        self.gradients = [torch.empty_like(p) for p in self.params]

    def check(self):
        from threading import get_ident

        if (
            self.thread != get_ident()
            or self.caller_stream != torch.cuda.current_stream(self.params[0].device)
        ):
            raise RuntimeError(
                "Shared workspaces must replay on their original thread and stream"
            )

    def publish(self, outputs):
        # No graph-private prediction or gradient is carried across replays.
        gradients = [p.grad for p in self.params]
        if any(g is None for g in gradients):
            raise RuntimeError("Shared training workspaces expect all parameter gradients")
        torch._foreach_copy_(self.gradients, gradients)
        for p, g in zip(self.params, self.gradients, strict=True):
            p.grad = g
        return {k: v.clone() for k, v in outputs.items()}


class MultiBucketEFSTrainingStep:
    """Train a supported foundation model with the smallest CUDA Graph that fits a batch.

    Parameters
    ----------
    model : torch.nn.Module
        Training-mode L, 7M or TECE-RRA adapter with energy and force targets,
        optionally stress. OAM and OMat24 checkpoints use the same architecture.
    optimizer : torch.optim.Optimizer
        Optimizer shared by all buckets. Each call performs one update.
    capacities : iterable of (int, int)
        Maximum real atom and directed-edge counts per bucket. All batches
        must contain the same number of structures. Among matching capacities,
        the one with the fewest edges, then fewest atoms, is selected.
    joint : bool, optional
        Use the joint derivative program in each bucket.
    share_workspace : bool, optional
        Share temporary storage across serial, non-joint fused AdamW steps.
        Use one calling thread and CUDA stream. Model, gradients and optimizer
        state remain outside the pool. Hooks and concurrent use are not supported.
    native_product : bool, optional
        Use native-layout quadratic products and their mixed derivatives for a
        specialized FP32 CUDA OAM-L model.

    Examples
    --------
    Given a supported CUDA model, its fused AdamW optimizer and example
    batches, prepare the shared workspace before the first update::

        caps = [(288, 10240), (448, 21760)]
        step = MultiBucketEFSTrainingStep(model, optimizer, caps, share_workspace=True)
        step.prepare_all(example_batches)
        predictions = step(batch)
    """

    def __init__(
        self,
        model,
        optimizer,
        capacities,
        *,
        joint=False,
        share_workspace=False,
        native_product=False,
    ):
        self.capacities = sorted(
            {(n, e) for n, e in capacities}, key=lambda c: (c[1], c[0])
        )
        if not self.capacities or any(
            type(n) is not int or type(e) is not int or n < 1 or e < 0
            for n, e in self.capacities
        ):
            raise ValueError(
                "Expected one or more (positive atoms, nonnegative edges) capacities"
            )
        self.model, self.optimizer, self.joint = model, optimizer, joint
        self._steps = {}
        self.graphs = None
        self.workspace = (
            _SerialTrainingWorkspace(model, optimizer, joint)
            if share_workspace
            else None
        )
        self._native_products = ()
        if native_product:
            from .tace_oam_l.model import _OAMLRepresentation
            from .tace_oam_l.product_kernels import square_metadata

            rep = getattr(getattr(model, "readout_fn", None), "representation", None)
            if (
                not isinstance(rep, _OAMLRepresentation)
                or not model.training
                or rep.model_dtype != torch.float32
                or any(
                    not p.is_cuda or p.dtype != torch.float32
                    for p in rep.products.parameters()
                )
            ):
                raise ValueError(
                    "native_product requires a training-mode FP32 CUDA OAM-L model"
                )
            self._native_products = tuple(
                (product, square_metadata(product))
                for product in rep.products
                if not product.scalar_norms
            )

    def prepare(self, batch):
        """Capture the matching workspace on first use, without an optimizer update."""
        step = self._select(batch)
        return step if self.workspace is None else self

    def prepare_all(self, batches):
        """Prepare matching buckets without updating parameters.

        Parameters
        ----------
        batches : iterable of batches
            Example batches with the same number of structures. With a shared
            workspace all new buckets warm up before their graphs are recorded.

        Returns
        -------
        MultiBucketEFSTrainingStep
            This step. Unseen buckets are still prepared on first use.
        """
        if self.workspace is None:
            for batch in batches:
                self.prepare(batch)
            return self
        self.workspace.check()
        previous, graphs = self._steps.copy(), self.graphs
        recordings = []
        try:
            for batch in batches:
                self._select(batch, recordings)
            for record in recordings:
                with self._capture_state():
                    record()
        except BaseException:
            self._steps, self.graphs = previous, graphs
            raise
        return self

    @contextmanager
    def _capture_state(self):
        params = self.workspace.params if self.workspace is not None else ()
        saved = [p.grad for p in params]
        previous = [p.native_square for p, _ in self._native_products]
        try:
            for product, meta in self._native_products:
                product.native_square = meta
            yield
        finally:
            for (product, _), old in zip(self._native_products, previous):
                product.native_square = old
            for p, grad in zip(params, saved, strict=True):
                p.grad = grad

    def _select(self, batch, recordings=None):
        if self.workspace is not None:
            self.workspace.check()
        graphs = len(batch["ptr"]) - 1
        if self.graphs is not None and graphs != self.graphs:
            raise ValueError("The number of real structures per training batch changed")
        shape = (len(batch["positions"]), batch["edge_index"].shape[1])
        for cap in self.capacities:
            if shape[0] <= cap[0] and shape[1] <= cap[1]:
                if cap not in self._steps:
                    options = (
                        {} if self.workspace is None else {"_workspace": self.workspace}
                    )
                    if recordings is not None:
                        options["_recordings"] = recordings
                    with self._capture_state():
                        self._steps[cap] = BucketedEFSTrainingStep(
                            self.model,
                            self.optimizer,
                            batch,
                            max_atoms=cap[0],
                            max_edges=cap[1],
                            joint=self.joint,
                            **options,
                        )
                    self.graphs = graphs
                return self._steps[cap]
        raise ValueError(f"Batch {shape} exceeds configured training buckets")

    def __call__(self, batch):
        result = self._select(batch)(batch)
        return result if self.workspace is None else self.workspace.publish(result)
