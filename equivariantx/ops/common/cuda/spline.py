"""Fixed cubic splines without an edge-by-coefficients workspace."""

from .pointwise import evaluate as pointwise


def evaluate(x, coefficients, step, order):
    n, _, width = coefficients.shape
    output = x.new_empty((x.shape[0], width))
    if not output.numel():
        return output
    polynomial = (
        "((c[0]*dx+c[width])*dx+c[2*width])*dx+c[3*width]",
        "(T(3)*c[0]*dx+T(2)*c[width])*dx+c[2*width]",
        "T(6)*c[0]*dx+T(2)*c[width]",
        "T(6)*c[0]",
    )[order]
    body = f"""
    constexpr I width = {width}, intervals = {n};
    constexpr T step = T({float(step).hex()});
    T r = p0[i / width];
    if (!(r >= T(0) && r < intervals * step)) {{ out[i] = T(0); continue; }}
    I k = I(r / step);
    if (k >= intervals) k = intervals - 1;
    T dx = r - T(k) * step;
    const T* c = p1 + (4*k)*width + i % width;
    out[i] = {polynomial};
    """
    return pointwise(output, (x.contiguous(), coefficients.contiguous()), body)
