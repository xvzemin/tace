"""Demonstrate why pair data do not uniquely determine magnetic one-body energy.

The synthetic collinear model is

    E(m1, m2) = U(m1) + U(m2) + J m1 m2,
    U(m) = W[0] m**2 + W[1] m**4.

Run with::

    python example/magnetic_onebody_demo.py --output /tmp/onebody.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TRUE_W = np.array([0.40, 0.08])
TRUE_J = -2.00
TRUE_PARAMETERS = np.append(TRUE_W, TRUE_J)


def features(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Linear features multiplying ``[W[0], W[1], J]``."""
    return np.column_stack((m1**2 + m2**2, m1**4 + m2**4, m1 * m2))


def one_body(m: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weights[0] * m**2 + weights[1] * m**4


def magnetic_force(m: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the one-body restoring force ``-dU/dm``."""
    return -2.0 * weights[0] * m - 4.0 * weights[1] * m**3


def fit(design: np.ndarray, energy: np.ndarray) -> tuple[np.ndarray, int]:
    parameters, _, rank, _ = np.linalg.lstsq(design, energy, rcond=None)
    return parameters, rank


def nonzero_minimum(weights: np.ndarray) -> float | None:
    w2, w4 = weights
    return float(np.sqrt(-w2 / (2.0 * w4))) if w2 < 0.0 < w4 else None


def main(output: Path) -> None:
    moments = np.linspace(-2.2, 2.2, 81)

    # Ferromagnetic pair data: m1 = m2 = m. Here the W[0] and J columns are
    # proportional, so only the combination 2 W[0] + J can be learned.
    pair_design = features(moments, moments)
    pair_energy = pair_design @ TRUE_PARAMETERS
    pair_fit, pair_rank = fit(pair_design, pair_energy)

    # Isolated-atom fixed-spin-moment (FSM) data contain no interaction term.
    isolated_design = features(moments, np.zeros_like(moments))
    isolated_energy = isolated_design @ TRUE_PARAMETERS
    joint_design = np.vstack((pair_design, isolated_design))
    joint_energy = np.concatenate((pair_energy, isolated_energy))
    constrained_fit, joint_rank = fit(joint_design, joint_energy)

    # Global time reversal only duplicates the FM equations; it adds symmetry,
    # but no information that separates one-body and exchange contributions.
    time_reversed_rank = np.linalg.matrix_rank(
        np.vstack((pair_design, features(-moments, -moments)))
    )

    grid = np.linspace(-2.4, 2.4, 401)
    pair_grid = features(grid, grid)
    styles = (
        (TRUE_PARAMETERS, "true decomposition", "-"),
        (pair_fit, "fit to FM pairs only", "--"),
        (constrained_fit, "FM pairs + isolated FSM", ":"),
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    for parameters, label, linestyle in styles:
        axes[0].plot(
            grid,
            pair_grid @ parameters,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            grid,
            one_body(grid, parameters[:2]),
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axes[2].plot(
            grid,
            magnetic_force(grid, parameters[:2]),
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )

    false_minimum = nonzero_minimum(pair_fit[:2])
    if false_minimum is not None:
        axes[1].scatter(
            [-false_minimum, false_minimum],
            one_body(np.array([-false_minimum, false_minimum]), pair_fit[:2]),
            marker="x",
            s=45,
            color="black",
            zorder=4,
            label="spurious isolated minima",
        )

    titles = (
        "FM pair energy",
        "Inferred isolated one-body energy",
        r"One-body magnetic force $-\mathrm{d}U/\mathrm{d}m$",
    )
    ylabels = ("energy", "energy", "magnetic force")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.axhline(0.0, color="0.75", linewidth=0.8)
        axis.axvline(0.0, color="0.75", linewidth=0.8)
        axis.set(title=title, xlabel="signed moment m", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    np.set_printoptions(precision=3, suppress=True)
    print(f"true W                        = {TRUE_W}")
    print(f"true J                        = {TRUE_J:.3f}")
    print(f"FM-only W                     = {pair_fit[:2]}")
    print(f"FM-only J                     = {pair_fit[2]:.3f}")
    print(f"FM-only design rank           = {pair_rank}/3")
    print(f"rank after time-reversed copy = {time_reversed_rank}/3")
    max_pair_error = np.max(np.abs(pair_design @ pair_fit - pair_energy))
    print(f"max FM training error         = {max_pair_error:.2e}")
    print(f"FM + isolated FSM W           = {constrained_fit[:2]}")
    print(f"FM + isolated FSM J           = {constrained_fit[2]:.3f}")
    print(f"joint design rank             = {joint_rank}/3")
    print(f"spurious isolated |m| minimum = {false_minimum:.3f}")
    print(f"figure saved to               = {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("magnetic_onebody_demo.png")
    )
    main(parser.parse_args().output)
