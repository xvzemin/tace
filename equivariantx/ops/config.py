"""Internal execution switches, read before operator dispatch."""

# Keep the ordinary forward program independent of the optimized adjoints.
TORCH_REFERENCE = False

# Recompute expression intermediates when activation memory is the priority.
TORCH_RECOMPUTE = False

# Bound per-element packing without reading data-dependent counts on the host.
TORCH_GROUPED_LINEAR = True

# Keep wide CG channel contractions in e3nn's measured GEMM schedule.
TORCH_GROUPED_CG = True

# Expose low-degree transverse convolutions before differentiation and compilation.
TRANSVERSE_POLYNOMIAL_DEGREE = 2

# Expose low-degree Cartesian channel products to the joint Torch compiler.
CARTESIAN_POLYNOMIAL_DEGREE = 2

# Expose measured small-order rotary primals to joint compilation.
ROTARY_INLINE = True
