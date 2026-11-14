"""Lazy import aliases for modules stored in older Python checkpoints."""

import importlib
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys


class KernelAlias(Loader):
    """Load the canonical module without repeating operator registration."""

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class KernelAliases(MetaPathFinder):
    """Resolve the former internal namespace only after it is requested."""

    def find_spec(self, fullname, path=None, target=None):
        prefix = "equivariantx.ops.kernels."
        if not fullname.startswith(prefix):
            return None
        canonical = "equivariantx.ops.common." + fullname[len(prefix) :]
        spec = find_spec(canonical)
        if spec is None:
            return None
        return spec_from_loader(
            fullname,
            KernelAlias(canonical),
            is_package=spec.submodule_search_locations is not None,
        )


sys.meta_path.insert(0, KernelAliases())
