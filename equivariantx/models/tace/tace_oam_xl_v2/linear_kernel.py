"""Native matrix products for grouped irrep channel maps."""


def run_program(metadata, program, operands):
    """Evaluate channel maps and their adjoints using strided matrix products."""
    from equivariantx.utils.metadata import parse_metadata

    from .linear import linear_groups, linear_outputs

    layout_x, layout_y, _, _, _ = parse_metadata(metadata)
    results = linear_outputs(metadata, program, operands)
    for result in results:
        result.zero_()

    def view(value, start, channels, dim, layout):
        block = value[:, start : start + channels * dim]
        return (
            block.reshape(value.shape[0], dim, channels)
            if layout == "ir_mul"
            else block.reshape(value.shape[0], channels, dim).transpose(1, 2)
        )

    for role, group in enumerate(linear_groups(metadata, program)):
        for (ix, ci, iy, co, dim, scale), slot, terms in group:
            for x, w, y in terms:
                if role == 2:
                    value = view(operands[x], ix, ci, dim, layout_x) @ operands[w]
                    view(results[slot], iy, co, dim, layout_y).add_(value, alpha=scale)
                elif role == 0:
                    value = view(operands[y], iy, co, dim, layout_y) @ operands[w].T
                    view(results[slot], ix, ci, dim, layout_x).add_(value, alpha=scale)
                else:
                    a = view(operands[x], ix, ci, dim, layout_x).reshape(-1, ci)
                    b = view(operands[y], iy, co, dim, layout_y).reshape(-1, co)
                    results[slot].addmm_(a.T, b, alpha=scale)
    return results
