"""Replay the complete compiled E/F/S loss and gradient program.

Parameters and optimizer matrices retain their original addresses. Input
buffers belong to each shape bucket; temporary allocations share one graph
pool. The optimizer and all training state updates remain outside capture.
"""

from collections import OrderedDict
from contextlib import nullcontext, contextmanager
import gc
import os

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

    def __init__(self, adapter, loss_fn, workspace, maximum=256, shared_graph_plan=True):
        self.plan_scope = nullcontext
        if shared_graph_plan:
            from .graph import graph_plan_scope
            self.plan_scope = graph_plan_scope
        self.adapter = adapter
        self.loss_fn = loss_fn
        self.properties = [key for key in loss_fn.loss_property if key in ("stress", "virials")]
        self.workspace = workspace
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        self.entries = OrderedDict()
        self.maximum = maximum
        self.warmed = {}
        self.captures = 0
        self.programs = {}
        if maximum < 1:
            raise ValueError("The graph cache must hold at least one entry")

    def __call__(self, batch):
        from tace.utils.loss.common import prepare_tensor_loss_counts
        from tace.models.compile.joint import JointTraining
        from torch._inductor import config
        from .precision import validate_call
        from .program import MixedGradientPass
        from .extensions import extension_identity

        validate_call(batch)
        extension_identity()
        batch = prepare_tensor_loss_counts(self.adapter.prepare_property_data(batch), self.properties)
        signature = tuple((k, tuple(v.shape), v.dtype, tuple(v.stride()))
                          if torch.is_tensor(v) else (k, repr(v))
                          for k, v in sorted(batch.items()))
        if signature not in self.entries:
            # Host-side loss-count preparation remains outside the captured
            # program, including its distributed count reduction when enabled.
            # Compile directly from already prepared counts. Calling the public
            # forward_loss here would conditionally run another all_reduce and
            # deadlock ranks observing different new capacities. Final partial
            # batches get their own structure-count contract; none are dropped.
            contract = (batch["ptr"].numel(), tuple(sorted(batch)),
                        min(batch["positions"].shape[0], 2),
                        min(batch["edge_index"].shape[1], 2))
            if contract not in self.programs:
                passes = {"post_grad_custom_pre_pass": MixedGradientPass()}
                if os.environ.get("EQX_XL_PARALLEL_MATRICES", "0") == "1":
                    passes["pre_grad_custom_pass"] = MixedGradientPass()
                with config.patch(passes):
                    self.programs[contract] = JointTraining(self.adapter, self.loss_fn, batch)
            program = self.programs[contract]
            static = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
            policy=(contract, batch["positions"].shape[0]>=64,batch["edge_index"].shape[1]>=1024)
            if policy not in self.warmed:
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    warm = program(static)
                torch.cuda.current_stream().wait_stream(self.stream)
                spec=({k:(tuple(v.shape),v.dtype) for k,v in warm[0].items()},warm[1].dtype)
                del warm
                self.warmed[policy] = spec
            spec = self.warmed[policy]
            outputs={}
            for key,(shape,dtype) in spec[0].items():
                shape=list(shape)
                if key in ("forces","node_energy"):shape[0]=batch["positions"].shape[0]
                elif key in ("energy","stress","virials"):shape[0]=batch["ptr"].numel()-1
                else:raise ValueError(f"Unsupported captured property: {key}")
                outputs[key]=torch.empty(shape,dtype=dtype,device=batch["positions"].device)
            loss_buffer=torch.empty((),dtype=spec[1],device=batch["positions"].device)
            if os.environ.get('EQX_XL_LT_RADIAL','0') == '1':
                from equivariantx.models.tace.tace_oam_xl_v2.lt_matrix import warm_capacity
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    warm_capacity(batch['edge_index'].shape[1],batch['positions'].device)
                torch.cuda.current_stream().wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with capture_lifetime(), torch.cuda.graph(graph, stream=self.stream, pool=self.pool), self.plan_scope():
                predictions, loss = program(static)
                self.workspace.attach_loss(loss)
                mask=tuple(p.grad is not None for p in self.workspace.parameters)
                torch._foreach_copy_(list(outputs.values()),[predictions[k] for k in outputs])
                loss_buffer.copy_(loss.detach())
                result=outputs,loss_buffer
            # Do not retain a separate 450 MB parameter-gradient output tuple
            # for every shape. All captures write the existing flat workspace.
            del loss,predictions
            self.entries[signature] = (static, graph, result, mask)
            self.captures += 1
            while len(self.entries) > self.maximum:
                self.entries.popitem(last=False)
        static, graph, result, mask = self.entries[signature]
        self.entries.move_to_end(signature)
        for key, value in batch.items():
            if torch.is_tensor(value):
                static[key].copy_(value)
        graph.replay()
        for parameter, view, active in zip(self.workspace.parameters,self.workspace.views,mask):
            parameter.grad=view if active else None
        return result

