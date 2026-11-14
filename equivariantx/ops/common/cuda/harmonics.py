"""Complex multiplication shared by real circular and Cartesian harmonics."""

import triton


@triton.jit
def complex_multiply(real, imaginary, x, y):
    return real * x - imaginary * y, real * y + imaginary * x
