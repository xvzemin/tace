"""Internal execution switches, read before operator dispatch."""

import torch

DEFAULT_BACKEND = "cuda" if torch.cuda.is_available() else "torch"

# Keep the ordinary forward program independent of the optimized adjoints.
TORCH_REFERENCE = False

# Inline analytic STF projection where surrounding Torch algebra can be fused.
TORCH_STF_PROJECTION = True

# Use signed planar reductions for measured ranks during joint compilation.
TORCH_PLANAR_PROJECTION = True
PLANAR_PROJECTION_INLINE = True
PLANAR_PROJECTION_NATIVE = False

# Retain operands instead of all Clenshaw intermediates.
TORCH_POLYNOMIAL_ADJOINT = True

# Save a bounded tape for eager circular orders >= 8; compiled algebra stays inline.
TORCH_CIRCULAR_ADJOINT = True

# Avoid full Cartesian powers for measured exact harmonic schedules.
TORCH_CARTESIAN_HARMONICS = True

# Evaluate measured CPU high-rank harmonics from exact monomials during capture.
TORCH_CARTESIAN_HARMONIC_POLYNOMIAL = True
TORCH_PLANAR_HARMONICS = True

# Expose standalone gates and static broadcast adjoints to joint compilation.
GATE_INLINE = True

# Recompute expression intermediates when activation memory is the priority.
TORCH_RECOMPUTE = False

# Save normalized values for eager attention; compiled graphs retain ordinary AD.
TORCH_ATTENTION_ADJOINT = True
TORCH_ATTENTION_MIN_CPU_ELEMENTS = 1 << 21

# Expose attention normalization before joint differentiation and compilation.
ATTENTION_INLINE = True
ATTENTION_NATIVE = False

# Shared channel-map operand-only tape; selected after performance measurement.
TORCH_CHANNEL_LINEAR = False

# Avoid batches of tiny GEMMs along short channel-preserving copy axes.
TORCH_UU_LINEAR = True

# Optional operand-only tape; ordinary explicit algebra is faster in joint graphs.
TORCH_UU_ADJOINT = False

# Expose short channelwise paths to Inductor before force-loss differentiation.
UU_LINEAR_INLINE = True

# Bound per-element packing without reading data-dependent counts on the host.
TORCH_GROUPED_LINEAR = True

# Merge matching CPU coefficient maps while retaining independent gradients.
TORCH_SHARED_COEFFICIENTS = True
SHARED_COEFFICIENTS_MAX_BYTES = 192 << 20

# Keep wide CG channel contractions in e3nn's measured GEMM schedule.
TORCH_GROUPED_CG = True
TORCH_CHANNEL_SLICES = True
CHANNEL_SLICES_MIN_WIDTH = 64

# Express fixed channel broadcasts as views and their adjoints as reductions.
TORCH_STATIC_COLUMNS = True

# Inline small angular polynomials into traced Torch graphs.
TORCH_PRODUCT_MAX_TERMS = 16
TORCH_BILINEAR_PRODUCT = True

# Expose low-degree transverse convolutions before differentiation and compilation.
TRANSVERSE_POLYNOMIAL_DEGREE = 2

# Expose low-degree Cartesian channel products to the joint Torch compiler.
CARTESIAN_POLYNOMIAL_DEGREE = 2

# Expose measured small-order rotary primals to joint compilation.
ROTARY_INLINE = True

# Share wide radial GEMMs before native spherical and aligned contractions.
# Zero disables this schedule; full-network fusion retains its tiled execution.
RADIAL_PROJECT_MIN_WIDTH = 1024
RADIAL_PROJECT_MAX_BYTES = 256 << 20

# Keep a packed affine bias outside wide, aligned radial GEMMs; zero disables it.
RADIAL_AFFINE_MIN_WIDTH = 1024

# Inline short tensor axes into joint graphs; eager Torch retains batched GEMMs.
TORCH_TENSOR_AXIS = True
TORCH_TENSOR_AXIS_ADJOINT = False

# Share Cartesian frame maps across wide channels during CPU compilation.
TORCH_CARTESIAN_FRAME_MATRIX = True
CARTESIAN_FRAME_MATRIX_MIN_CHANNELS = 64
TORCH_PLANAR_DETRACING = True
TENSOR_AXIS_INLINE = True

# Optional cubic polynomial trades mixed-gradient speed for less temporary storage.
TORCH_PLANAR_CUBIC = False

# Normalize compiled shared Cartesian GEMMs after contraction when paths agree.
TORCH_LINEAR_POSTSCALE = True

# Project shared rank-three CPU maps before wide channel contraction.
TORCH_COMPACT_LINEAR_BASIS_FIRST = True


# Direct normalized maps and sparse adjoints in jointly compiled symmetric transforms.
TORCH_SYMMETRIC_KERNELS = True
TORCH_SYMMETRIC_ADJOINT = True
TORCH_SYMMETRIC_MATRIX = True
SYMMETRIC_INLINE = True

# Reuse bounded ACE polynomial prefixes on CPU.
TORCH_SYMMETRIC_PREFIX = True
SYMMETRIC_PREFIX_MIN_WORK = 1 << 19
SYMMETRIC_PREFIX_MAX_TERMS = 256
SYMMETRIC_PREFIX_MAX_NODES = 1024

# Preserve native ACE calls in compiled graphs.
SYMMETRIC_NATIVE = False

# Optional shared representation matrices for measured CUDA channel widths.
SYMMETRIC_MATRIX_NATIVE = False

# Sparse generator actions; defaults are selected after component measurement.
TORCH_SPHERICAL_COUPLING = False
TORCH_SPHERICAL_ADJOINT = True
SPHERICAL_INLINE = False

# Share the direction-dependent coupling across feature channels.
SPHERICAL_MATRIX_MIN_CHANNELS = 16

# Reuse compact symmetric schedules and native tensor axes in transverse maps.
TORCH_RESTRICTION = True
RESTRICTION_INLINE = True

# Sparse Cartesian decomposition and exact transposed derivative programs.
TORCH_ICTD = True
TORCH_ICTD_ADJOINT = True
ICTD_INLINE = False
ICTD_FUSED_MAX_WIDTH = 128
ICTD_DENSE_MAX_WIDTH = 1024

# Compact transverse products and staged Cartesian contractions.
TORCH_TRANSVERSE_PRODUCT = True
TORCH_ICTP_PRODUCT = True
ICTP_INLINE = True

# Reorder measured wide CPU Cartesian outer products before angular expansion.
TORCH_CARTESIAN_REORDER = True
CARTESIAN_REORDER_MIN_WORK = 1 << 18

# Move sheet mixing to compact Fourier coefficients; test S2 parity separately.
TORCH_O2_GRID = True
O2_GRID_INLINE = False
TORCH_S2_GRID = False
S2_GRID_INLINE = False

# Explicit O(2) scalar rotations, couplings and shared-prefix contractions.
TORCH_ALIGNMENT_ADJOINT = False
ALIGNMENT_INLINE = False
WIGNER_INLINE = True
TORCH_O2_COUPLING = True
TORCH_O2_COUPLING_ADJOINT = True
O2_COUPLING_INLINE = False
TORCH_O2_CONTRACTION = False
TORCH_O2_CONTRACTION_ADJOINT = True
O2_CONTRACTION_STAGED = True
O2_CONTRACTION_EAGER_STAGED = False
O2_CONTRACTION_INLINE = True

# Share channel contractions across aligned magnetic components.
TORCH_ALIGNED_PRODUCT = True
ALIGNED_PRODUCT_INLINE = False

# Exact compact angular contractions for low-degree factorized products.
TORCH_FACTORIZED_PRODUCT = True
FACTORIZED_INLINE = False

# Share first projections before Wigner-6j recoupling.
TORCH_WIGNER6J = True
TORCH_WIGNER6J_ADJOINT = False
WIGNER6J_INLINE = False
WIGNER6J_RECOUPLE = None

# Expose existing programs to compilation; eager execution retains native kernels.
EDGE_PROGRAM_INLINE = True
POLYNOMIAL_INLINE = True

# Keep compiler-visible indexed maps optional until wider workloads are measured.
LINEAR_NATIVE = False
