"""Independent matrix-basis evaluation for spherical products."""


def transform(module, features, basis):
    return (features @ basis).reshape(
        features.shape[:-1] + (module.matrix_dim, module.matrix_dim)
    )


def product(module, x, y, name):
    return (x @ y).flatten(-2) @ getattr(module, f"{name}_basis")
