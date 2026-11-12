"""Tests for the corresponding EQX source module."""

import equivariantx
import ast
import doctest
from pathlib import Path
import torch
from equivariantx import o2
from equivariantx import o3 as eqx_o3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPACES = {2: o2, 3: eqx_o3}
ROOT = Path(equivariantx.__file__).resolve().parent


def has_documentation_examples(path):
    """Collect executable examples, excluding commented usage sketches."""
    parser = doctest.DocTestParser()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if parser.get_examples(ast.get_docstring(node) or ""):
                return True
    return False
