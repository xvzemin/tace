"""Tests for TACE-specific gated product kernels."""

from equivariantx_foundation.models.tace.bilinear import BilinearContraction
from equivariantx_foundation.models.tace.ictd_bilinear import IctdBilinearContraction

GATED = {"o3": BilinearContraction, "co3": IctdBilinearContraction}
