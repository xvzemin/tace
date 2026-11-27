"""Bound neighbors per CTA while retaining every mixed-gradient contribution.

Only explicit source/mixed adjoints opt in. Radial outputs keep exclusive
ownership; node sums use FP32/FP64 atomics across disjoint row segments.
"""

import re

import torch


def split_source(code, cap, receiver=False):
    code, count = re.subn(
        r'(extern "C" __global__ void run\([^)]*)\)',
        r"\1, const int64_t* jobs)",
        code,
        count=1,
    )
    if count != 1:
        raise ValueError("Kernel signature not found")
    if receiver:
        code = code.replace(
            "const int64_t node = blockIdx.x;", "const int64_t task=blockIdx.x;"
        )
        code = code.replace(
            "if (node >= tasks) return;", "if (task >= jobs[tasks]) return;"
        )
    else:
        code = code.replace(
            "if (task >= tasks) return;", "if (task >= jobs[tasks]) return;"
        )
        code = code.replace("const int64_t node=task;", "")
    marker = "const int64_t begin=pointers[node], end=pointers[node+1];"
    if code.count(marker) != 1:
        raise ValueError("Expected one full-row traversal")
    code = code.replace(
        marker,
        "int64_t lower=0, upper=tasks;\n"
        "while(lower<upper){int64_t middle=(lower+upper)/2;"
        "if(jobs[middle+1]<=task) lower=middle+1; else upper=middle;}\n"
        "const int64_t node=lower;\n"
        f"const int64_t begin=pointers[node]+(task-jobs[node])*{cap};\n"
        f"const int64_t end=min(begin+{cap},pointers[node+1]);",
    )
    if receiver:
        code, count = re.subn(
            r"if \(active\) (g\d+\[[^\]]+\]) = (sum\d+);",
            r"if (active) atomicAdd(&\1, \2);",
            code,
        )
        if not count:
            raise ValueError("Receiver owned sums not found")
    else:
        code = code.replace("const bool exclusive=true;", "const bool exclusive=false;")
    return code


def prepare_segments(pointers, cap):
    """Build a live task prefix, including empty nodes, on the current stream."""
    if cap <= 0:
        raise ValueError("Segment length must be positive")
    counts = torch.div(
        pointers[1:] - pointers[:-1] + cap - 1, cap, rounding_mode="floor"
    )
    return torch.cat((pointers.new_zeros(1), counts.cumsum(0)))
