################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import copy
import gc

import torch

from equivariantx_foundation.models.tace.tace_oam_xl_v2 import TACEOAMXLV2
from equivariantx_foundation.models.tace.tace_oam_xl_v2.capture import CapturedTraining
from equivariantx_foundation.models.tace.tace_oam_xl_v2.padding import (
    pad_batch,
    padded_force_loss,
)
from equivariantx_foundation.models.tace.tace_oam_xl_v2.precision import (
    require_tensors,
    validate_environment,
)
from equivariantx_foundation.models.tace.tace_oam_xl_v2.training import (
    GradientWorkspace,
)
from tace.lightning.lit_model import LightningWrapperModel
from tace.utils.env import memory_first


class OAMLightningModule(LightningWrapperModel):
    """Use the retained FP32 schedule with canonical TACE checkpoint state."""

    def __init__(self, *args, capture_limit=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.capture_limit = capture_limit
        self.captured_training = None
        object.__setattr__(self, "canonical_model", None)
        self.workspace = None
        self.training_loss = None

    def configure_optimizers(self):
        return super().configure_optimizers(shard_muon=True)

    def training_step(self, batch, batch_idx):
        """Consume the complete E/F/S parameter gradients from the joint graph."""
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        loss = self._shared_step(batch, batch_idx, "train")
        self.workspace.synchronize()
        optimizer.step()
        self.step_scheduler("step", self.global_step)
        return loss.detach()

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.step_scheduler("epoch", self.current_epoch + 1)

    def step_scheduler(self, interval, index):
        for schedule in self.trainer.lr_scheduler_configs:
            if schedule.interval == interval and index % schedule.frequency == 0:
                if isinstance(
                    schedule.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    monitor = (
                        self.cfg["scheduler"]
                        .get("extra", {})
                        .get("monitor", "val/loss")
                    )
                    metric = self.trainer.callback_metrics.get(monitor)
                    if metric is not None:
                        schedule.scheduler.step(metric)
                else:
                    schedule.scheduler.step()

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        maximum = self.cfg["trainer"].get("gradient_clip_val")
        if maximum:
            algorithm = self.cfg["trainer"].get("gradient_clip_algorithm", "norm")
            self.configure_gradient_clipping(optimizer, maximum, algorithm)

    def prepare_execution(self):
        """Convert after checkpoint restoration, retaining Parameter identities."""
        if self.canonical_model is not None:
            return
        validate_environment()
        if (
            type(self.loss_fn).__name__ != "NormalLoss"
            or next(self.loss_fn.parameters(), None) is not None
        ):
            raise ValueError(
                "This production entry uses the original parameter-free NormalLoss"
            )
        require_tensors((*self.model.parameters(), *self.model.buffers()))
        original = self.model
        from equivariantx_foundation.models.tace.tace_oam_xl_v2.radial_jet import (
            supports_degree_jets,
        )

        stream = (
            memory_first()
            and original.readout_fn.representation.radial_basis.widths is not None
        )
        from equivariantx_foundation import optimize

        self.model = optimize(
            original,
            family="tace",
            task="joint_training",
            backend="cuda",
            fuse_products=False,
            radial_mode="stream" if stream else "materialize",
            angular_mode="cg",
            product_layout="ir_mul",
            linear_mode="torch",
            edge_mode="original" if stream else "scheduled",
            gradient_schedule="fused",
            product_mode="packed",
            compose_products=True,
            affine_radial=True,
            padded_batches=True,
            radial_jets=(
                not stream
                and set(self.loss_fn.loss_property)
                <= {"energy", "forces", "stress", "virials"}
                and supports_degree_jets(original.readout_fn.representation)
            ),
        )
        object.__setattr__(self, "canonical_model", original)
        self.training_loss = padded_force_loss(copy.deepcopy(self.loss_fn))

    def _process_batch(self, batch):
        self.prepare_execution()
        # Lightning restores registered training flags without calling train().
        # The unregistered adapter must follow the active loop before tracing.
        if self.model.adapter.training != self.training:
            self.model.train(self.training)
            self.training_loss.train(self.training)
        if not self.training:
            return super()._process_batch(batch)
        if self.captured_training is None:
            self.workspace = GradientWorkspace(
                p for p in self.model.parameters() if p.requires_grad
            )
            self.captured_training = CapturedTraining(
                self.model.adapter,
                self.training_loss,
                self.workspace,
                maximum=self.capture_limit,
                shared_graph_plan=True,
            )
            self.captured_training.programs = self._training_programs()
        data = dict(batch.items())
        nodes = data["positions"].shape[0]
        predictions, value = self.captured_training(pad_batch(data, 64, 1024))
        loss = value.detach()
        predictions = {
            key: tensor[:nodes] if key in ("forces", "node_energy") else tensor
            for key, tensor in predictions.items()
        }
        return predictions, loss

    def _training_programs(self):
        # The adapter invalidates its cache on device, dtype, target or fidelity
        # changes. Optimizer updates and temporary EMA weights are live inputs.
        cache = self.model.adapter._compiled_cache
        loss = self.training_loss
        key = (
            id(loss),
            tuple(loss.loss_property),
            tuple(loss.loss_function_name),
            tuple(loss.loss_property_weights),
            repr(loss.loss_function_kwargs),
            tuple((id(p), p.requires_grad) for p in self.model.parameters()),
        )
        cached = cache.get("oam_training_programs")
        if cached is None or cached[0] != key:
            cached = cache["oam_training_programs"] = (key, {})
        return cached[1]

    def release_training_capture(self, *, keep_programs=False):
        """Release graph storage and its gradient views before evaluation."""
        if not keep_programs and self.canonical_model is not None:
            self.model.adapter._compiled_cache.pop("oam_training_programs", None)
        if self.captured_training is None:
            return
        torch.cuda.synchronize(self.captured_training.stream.device)
        if keep_programs:
            programs = self.captured_training.programs
            reusable = [
                (key, program)
                for key, program in programs.items()
                if program.artifact_info["status"] in ("hit", "saved")
            ][-self.capture_limit :]
            programs.clear()
            programs.update(reusable)
        self.workspace.detach()
        self.captured_training = None
        self.workspace = None
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_start(self):
        from lightning.pytorch.trainer.states import TrainerFn

        self.release_training_capture(
            keep_programs=self.trainer.state.fn == TrainerFn.FITTING
        )

    def on_train_end(self):
        self.release_training_capture()

    def on_test_start(self):
        self.release_training_capture()

    def on_validation_end(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def on_test_end(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def on_save_checkpoint(self, checkpoint):
        """Export original-layout buffers with current, shared trainable weights."""
        if self.canonical_model is not None:
            state = checkpoint["state_dict"]
            for key in list(state):
                if key.startswith("model."):
                    del state[key]
            state.update(
                ("model." + key, value)
                for key, value in self.canonical_model.state_dict().items()
            )
        checkpoint["oam_execution"] = {
            "format": 1,
            "model_precision": "fp32/fp64/tf32",
            "model_name": TACEOAMXLV2.name,
            "canonical_tace_state": True,
            "capture_limit": self.capture_limit,
        }

    def on_load_checkpoint(self, checkpoint):
        """Restore canonical state before converting again, also for trainer.test."""
        if self.canonical_model is not None:
            self.release_training_capture()
            self.model = self.canonical_model
            object.__setattr__(self, "canonical_model", None)
            self.training_loss = None
