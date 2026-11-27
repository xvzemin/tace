"""Tests for TACE-specific gated product kernels."""

from tace.ops.bilinear import BilinearContraction
from tace.ops.ictd_bilinear import IctdBilinearContraction

GATED = {"o3": BilinearContraction, "co3": IctdBilinearContraction}
