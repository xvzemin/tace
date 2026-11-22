"""Replay the complete compiled E/F/S loss and gradient program.

Parameters and optimizer matrices retain their original addresses. Input
buffers belong to each shape bucket; temporary allocations share one graph
pool. The optimizer and all training state updates remain outside capture.
"""

import gc
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def capture_lifetime():
    """Keep cyclic GC from unloading compiled CUDA modules during capture."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class CapturedTraining:
    """Capture one explicit joint program per observed input signature."""

    def __init__(
        self, adapter, loss_fn, workspace, maximum=256, shared_graph_plan=True
    ):
        self.plan_scope = nullcontext
        if shared_graph_plan:
            from .graph import graph_plan_scope

            self.plan_scope = graph_plan_scope
        self.adapter = adapter
        self.loss_fn = loss_fn
        self.properties = [
            key for key in loss_fn.loss_property if key in ("stress", "virials")
        ]
        self.workspace = workspace
        self.memory_pool = torch.cuda.MemPool()
        self.pool = self.memory_pool.id
        self.stream = torch.cuda.Stream()
        self.entries = OrderedDict()
        self.maximum = maximum
        self.warmups = 0
        self.pool_resets = 0
        self.captures = 0
        self.programs = {}
        if maximum < 1:
            raise ValueError("The graph cache must hold at least one entry")

    def __call__(self, batch):
        from torch._inductor import config

        from tace.models.compile.joint import JointTraining
        from tace.utils.loss.common import prepare_tensor_loss_counts

        from .extensions import extension_identity
        from .precision import validate_call
        from .program import MixedGradientPass

        validate_call(batch)
        extension_identity()
        # The distributed loss counts must be prepared exactly once per step,
        # including ranks whose graph signature is already cached.
        batch = prepare_tensor_loss_counts(
            self.adapter.prepare_property_data(batch), self.properties
        )
        signature = tuple(
            (key, tuple(value.shape), value.dtype, tuple(value.stride()))
            if torch.is_tensor(value)
            else (key, repr(value))
            for key, value in sorted(batch.items())
        )
        if signature not in self.entries:
            contract = (
                batch["ptr"].numel(),
                tuple(sorted(batch)),
                min(batch["positions"].shape[0], 2),
                min(batch["edge_index"].shape[1], 2),
            )
            if contract not in self.programs:
                passes = {"post_grad_custom_pre_pass": MixedGradientPass()}
                if os.environ.get("EQX_XL_PARALLEL_MATRICES", "0") == "1":
                    passes["pre_grad_custom_pass"] = MixedGradientPass()
                with config.patch(passes):
                    self.programs[contract] = JointTraining(
                        self.adapter, self.loss_fn, batch
                    )
            program = self.programs[contract]
            static = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            device = batch["positions"].device
            current = torch.cuda.current_stream(device)
            self.stream.wait_stream(current)
            for attempt in range(2):
                try:
                    # Warmup and capture share a pool, but older graphs pin its
                    # address ranges. A larger shape can need a fresh pool.
                    with (
                        capture_lifetime(),
                        torch.cuda.stream(self.stream),
                        torch.cuda.use_mem_pool(self.memory_pool, device),
                        self.plan_scope(),
                    ):
                        predictions, loss = program(static)
                        spec = {
                            key: (tuple(value.shape), value.dtype)
                            for key, value in predictions.items()
                        }
                        loss_dtype = loss.dtype
                        del predictions, loss
                    break
                except torch.OutOfMemoryError:
                    if attempt or not self.entries:
                        raise
                # Leave the exception handler first so its traceback releases
                # temporary tensors. Parameters and optimizer state stay live.
                torch.cuda.synchronize(device)
                self.entries.clear()
                gc.collect()
                self.memory_pool = torch.cuda.MemPool()
                self.pool = self.memory_pool.id
                torch.cuda.empty_cache()
                self.pool_resets += 1
                logging.info(
                    "Released cached CUDA Graphs for a larger training workspace"
                )
            self.stream.synchronize()
            self.warmups += 1
            outputs = {
                key: torch.empty(shape, dtype=dtype, device=device)
                for key, (shape, dtype) in spec.items()
            }
            loss_buffer = torch.empty((), dtype=loss_dtype, device=device)
            graph = torch.cuda.CUDAGraph()
            # DataLoader pin-memory threads allocate independently of this graph.
            with (
                capture_lifetime(),
                torch.cuda.graph(
                    graph,
                    stream=self.stream,
                    pool=self.memory_pool,
                    capture_error_mode="thread_local",
                ),
                self.plan_scope(),
            ):
                predictions, loss = program(static)
                self.workspace.attach_loss(loss)
                mask = tuple(
                    parameter.grad is not None
                    for parameter in self.workspace.parameters
                )
                torch._foreach_copy_(
                    list(outputs.values()), [predictions[key] for key in outputs]
                )
                loss_buffer.copy_(loss.detach())
            del predictions, loss
            self.entries[signature] = (static, graph, (outputs, loss_buffer), mask)
            self.captures += 1
            while len(self.entries) > self.maximum:
                self.entries.popitem(last=False)
        static, graph, result, mask = self.entries[signature]
        self.entries.move_to_end(signature)
        for key, value in batch.items():
            if torch.is_tensor(value):
                static[key].copy_(value)
        graph.replay()
        for parameter, view, active in zip(
            self.workspace.parameters, self.workspace.views, mask
        ):
            parameter.grad = view if active else None
        return result
