"""Avoid expanding large operator specifications into compiler comments."""

from contextlib import contextmanager
from importlib import import_module
from threading import RLock

compile_comment_lock = RLock()


@contextmanager
def omit_provenance_comments():
    """Skip descriptive Triton and external-call comments during compilation."""
    modules = {
        name: import_module(name)
        for name in ("torch._inductor.codegen.triton", "torch._inductor.ir")
    }
    with compile_comment_lock:
        originals = {
            name: module.get_kernel_metadata for name, module in modules.items()
        }

        def comments_only(*args, **kwargs):
            return "", ""

        try:
            for name, module in modules.items():
                module.get_kernel_metadata = comments_only
            yield
        finally:
            for name, module in modules.items():
                module.get_kernel_metadata = originals[name]
