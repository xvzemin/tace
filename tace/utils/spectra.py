################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import numpy as np
from scipy.fftpack import dct


from .units import c, hbar, Boltzmann


def ACF(x):
    n = len(x)
    return np.correlate(x, x, "full")[-n:] / np.arange(n, 0, -1)


def d_ACF(dipole, N: int):
    dacf = ACF(dipole[:, 0]) + ACF(dipole[:, 1]) + ACF(dipole[:, 2])
    dacf = dacf[:N]
    return dacf


def pGamma_ACF(polar, N: int):
    gamma = np.zeros(len(polar))
    for i in range(len(polar)):
        gamma[i] = np.trace(polar[i]) / 3
    pgammaacf = ACF(gamma)
    pgammaacf = pgammaacf[:N]
    return pgammaacf


def pBeta_ACF(polar, N: int):
    beta = np.zeros_like(polar)
    for i in range(len(polar)):
        beta[i] = polar[i] - np.eye(3) * np.trace(polar[i]) / 3
    pbetaacf = np.zeros(len(beta))
    for i in range(3):
        for j in range(3):
            pbetaacf += ACF(beta[:, i, j])
    pbetaacf = pbetaacf[:N]
    return pbetaacf


def IR(dacf, N: int, T: float, dt: float, w_max: float):
    ir = dct(dacf, type=1)
    freq = np.linspace(0, 0.5 / dt, N) / (100.0 * c)
    ir = freq**2 * ir
    n = int(w_max / freq.max() * N)
    return freq[:n], ir[:n]


def Raman(acf_beta, N, T, dt, w_max):
    raman = dct(acf_beta, type=1)
    freq = np.linspace(0, 0.5 / dt, N) / (100.0 * c)
    raman = freq**2 * raman
    n = int(w_max / freq.max() * N)
    return freq[:n], raman[:n]
