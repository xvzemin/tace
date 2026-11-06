from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hybrid_muon import HybridMuonOptimizer

__all__ = [
    "HybridMuonOptimizer",
]


def __getattr__(name):
    if name == "HybridMuonOptimizer":
        from .hybrid_muon import HybridMuonOptimizer

        globals()[name] = HybridMuonOptimizer
        return HybridMuonOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
