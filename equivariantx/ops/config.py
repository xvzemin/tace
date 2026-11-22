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

# Limit eager scratch per wide edge operand, independently of graph size.
TORCH_EDGE_TILE_BYTES = 128 << 20

# Save normalized values for eager attention; compiled graphs retain ordinary AD.
TORCH_ATTENTION_ADJOINT = True
TORCH_ATTENTION_MIN_CPU_ELEMENTS = 1 << 21

# Expose attention normalization before joint differentiation and compilation.
ATTENTION_INLINE = True
ATTENTION_NATIVE = False

# Shared channel-map operand-only tape; selected after performance measurement.
TORCH_CHANNEL_LINEAR = False
TORCH_O2_CHANNEL_LINEAR = True

# Pack the two components only for narrow FP32 channel maps.
TORCH_LINEAR_COMPONENT_MAX_CHANNELS = 64

# Avoid batches of tiny GEMMs along short channel-preserving copy axes.
TORCH_UU_LINEAR = True

# Optional operand-only tape; ordinary explicit algebra is faster in joint graphs.
TORCH_UU_ADJOINT = False

# Expose short channelwise paths to Inductor before force-loss differentiation.
UU_LINEAR_INLINE = True

# Bound per-element packing without reading data-dependent counts on the host.
TORCH_GROUPED_LINEAR = True
ELEMENT_LINEAR_WORKSPACE = 16 << 20

# Merge matching CPU coefficient maps while retaining independent gradients.
TORCH_SHARED_COEFFICIENTS = True
SHARED_COEFFICIENTS_MAX_BYTES = 192 << 20

# Keep wide CG channel contractions in e3nn's measured GEMM schedule.
TORCH_GROUPED_CG = True
TORCH_CG_ANGULAR_MATRICES = True
TORCH_CG_ANGULAR_WIDTH_RATIO = 24
# Reuse one packed rotation across low-degree, multi-path aligned convolutions.
TORCH_ALIGNED_MAX_DEGREE = 3
TORCH_CHANNEL_SLICES = True
CHANNEL_SLICES_MIN_WIDTH = 64

# Express fixed channel broadcasts as views and their adjoints as reductions.
TORCH_STATIC_COLUMNS = True

# Inline small angular polynomials into traced Torch graphs.
TORCH_PRODUCT_MAX_TERMS = 16
TORCH_BILINEAR_PRODUCT = True

# Expose low-degree transverse convolutions before differentiation and compilation.
TRANSVERSE_POLYNOMIAL_DEGREE = 2
UU_O2_SHARED_HARMONICS_DEGREE = 6
UU_O2_DENSE_WEIGHT_ORDERS = 32

# Expose low-degree Cartesian channel products to the joint Torch compiler.
CARTESIAN_POLYNOMIAL_DEGREE = 2
# Bound literal delta/epsilon programs independently of the number of channels.
CARTESIAN_LITERAL_TERMS = 1024
# Expose short contracted axes and scalar dot products to compiler fusion.
CARTESIAN_REDUCTION_INLINE = True
# Bound the eager Torch temporary for full scalar contractions.
TORCH_CARTESIAN_DOT_BYTES = 16 << 20

# Expose measured small-order rotary primals to joint compilation.
ROTARY_INLINE = True

# Share wide radial GEMMs before native spherical and aligned contractions.
# Zero disables this schedule; full-network fusion retains its tiled execution.
RADIAL_PROJECT_MIN_WIDTH = 1024
RADIAL_PROJECT_MAX_BYTES = 256 << 20
# Bound temporary storage for grouped radial networks.
RADIAL_CHAIN_TILE_BYTES = 512 << 20
# No-gradient pair MLPs can run directly if all their hidden rows fit.
PAIR_EAGER_MAX_BYTES = 64 << 20

# Tiny shared row maps can be cheaper as columns of the native final GEMM.
INDEXED_PROJECTION_MAX_ROWS = 8
# PyTorch benefits from wider packed GEMMs without a native indexed epilogue.
INDEXED_TORCH_PROJECTION_MAX_ROWS = 128
INDEXED_PROJECTION_MAP_BYTES = 8 << 20

# Keep a packed affine bias outside wide, aligned radial GEMMs; zero disables it.
RADIAL_AFFINE_MIN_WIDTH = 1024

# Inline short tensor axes into joint graphs; eager Torch retains batched GEMMs.
TORCH_TENSOR_AXIS = True
TORCH_TENSOR_AXIS_ADJOINT = False

# Share Cartesian frame maps across wide channels.
TORCH_CARTESIAN_FRAME_MATRIX = True
CUDA_CARTESIAN_FRAME_MATRIX = True
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

# Group wide CUDA-device Torch ACE polynomials. Keep their native CUDA counterpart during capture.
TORCH_SYMMETRIC_MONOMIAL_GEMM = True
SYMMETRIC_PROJECTION_MAX_ENTRIES = 1 << 18
SYMMETRIC_MONOMIAL_WORKSPACE = 64 << 20
TORCH_ACE_CHANNEL_POLYNOMIAL = True
ACE_POLYNOMIAL_WORKSPACE = 12 << 20
ACE_ELEMENT_BLOCK = 16
# Pack independent coefficient tables together during Torch graph capture.
TORCH_ACE_PACK_TABLES = True
# Limit node-by-channel-table expansion during native joint graph capture.
ACE_INLINE_MAX_ELEMENTS = 1 << 18
SYMMETRIC_WIDE_CHANNELS = 16
SYMMETRIC_WIDE_TERMS = 32
SYMMETRIC_SORT_MIN_NODES = 256

# Also preserve small ACE calls during joint tracing.
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
O2_GRID_FUSED = True
O2_GRID_MATRIX_ELEMENTS = 1 << 15
O2_GRID_MAX_SAMPLES = 96
O2_GRID_MAX_BANDS = 48
O2_GRID_TILE_BYTES = 64 << 20
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

# Retain CUDA calls for large programs and their expanded derivative schedules.
COMPILE_MAX_INLINE_WORK = 256

# Keep compiler-visible indexed maps optional until wider workloads are measured.
LINEAR_NATIVE = False
