"""Execution plans for TACE models."""

__all__ = ["plan"]


def __getattr__(name):
    # A compiled operator may live under this package without a host model.
    if name == "plan":
        from .planner import plan

        globals()[name] = plan
        return plan
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
