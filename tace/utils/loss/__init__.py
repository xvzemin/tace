###########################################################################################
# Authors: Zemin Xu
# License: Academic / Non-Commercial Use Only
###########################################################################################

import os
import glob
import importlib

__all__ = []
py_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
py_files = [f for f in py_files if not f.endswith("__init__.py")]

for p in py_files:
    module_name = os.path.splitext(os.path.basename(p))[0]
    module = importlib.import_module(f".{module_name}", package=__name__)
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type):
            globals()[attr_name] = attr
            __all__.append(attr_name)
