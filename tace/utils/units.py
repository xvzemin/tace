################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from scipy.constants import (
    pi,
    e,
    c,
    Planck,
    hbar,
    Boltzmann,
    epsilon_0,
    Avogadro,
    R,
    eV,
    angstrom,
    femto,
    pico,
    nano,
    bar,
    calorie,
)

kcalPerMol = 1000 * calorie / Avogadro


# === unit convert ===
kcalpermol2ev = kcalPerMol / eV
ev2kcalpermol = eV / kcalPerMol

pa2eva3 = (1 / eV) / (1e30)
kbar2eva3 = 1e8 * pa2eva3