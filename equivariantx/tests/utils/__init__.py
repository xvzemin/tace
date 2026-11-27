"""Shared numerical test utilities."""

import sys

import pytest
import torch


@pytest.fixture
def aoti_threads():
    """Use matching compiler/runtime thread counts in standalone CPU checks."""
    from torch._inductor import config

    previous = torch.get_num_threads()
    # Windows OpenMP workers can outlive an unloaded AOTI library. Keep these
    # standalone checks serial on Windows, as for the macOS runtime.
    threads = 1 if sys.platform in ("darwin", "win32") else min(previous, 4)
    torch.set_num_threads(threads)
    try:
        with config.patch({"cpp.threads": threads}):
            yield threads
    finally:
        torch.set_num_threads(previous)
