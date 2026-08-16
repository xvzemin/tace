import numpy as np
import scipy.special
import torch
from e3nn.o3 import spherical_harmonics

from .utils import permute_to_e3nn_convention


def old_tensor_get_grid_cell(Delta, cell):
    # use delta as the spacing for all the directions
    # cell is [3,3], first index is the lattice vector

    vector_norms = torch.sqrt(torch.sum(torch.square(cell), axis=1))
    normed_vectors = torch.divide(cell, vector_norms)
    g_spacing = normed_vectors * Delta
    data_shape = torch.round(
        torch.divide(
            vector_norms, torch.sqrt(torch.sum(torch.square(g_spacing), axis=1))
        )
    )

    origin = torch.zeros((3))

    # messy bit
    numpy_data_shape = data_shape.cpu().detach().numpy()
    coeficients = []
    for i in range(3):
        coeficients.append(np.arange(0, numpy_data_shape[i], 1, dtype=float))

    coefs = torch.from_numpy(
        np.asarray(
            np.meshgrid(coeficients[0], coeficients[1], coeficients[2], indexing="ij")
        )
    )
    coordinates = torch.einsum("iabc,ij->abcj", coefs, g_spacing)
    coordinates = coordinates + origin

    # [a,b,c,j], with j being the direction index
    return coordinates


def tensor_get_grid_cell(Delta, cell):
    # use delta as the approximate spacing in all directions
    # cell is [3,3], first index is the lattice vector
    # we use Delta as the approximate spacing

    vector_norms = torch.sqrt(torch.sum(torch.square(cell), axis=1))
    normed_vectors = torch.divide(cell, vector_norms)
    approx_spacing = normed_vectors * Delta
    data_shape = torch.floor(
        torch.divide(
            vector_norms, torch.sqrt(torch.sum(torch.square(approx_spacing), axis=1))
        )
    )

    g_spacing = torch.divide(cell, data_shape.unsqueeze(0))
    origin = torch.zeros((3))

    # messy bit
    numpy_data_shape = data_shape.cpu().detach().numpy()
    coeficients = []
    for i in range(3):
        coeficients.append(np.arange(0, numpy_data_shape[i] + 1, 1, dtype=float))

    coefs = torch.from_numpy(
        np.asarray(
            np.meshgrid(coeficients[0], coeficients[1], coeficients[2], indexing="ij")
        )
    )
    coordinates = torch.einsum("iabc,ij->abcj", coefs, g_spacing)
    coordinates = coordinates + origin

    # [a,b,c,j], with j being the direction index
    return coordinates


def tensor_realspace_GTO_evaluation(coordinates, l, sigma, center, normalize="none"):
    # the basis function is defined
    #   \phi_{nlm}(x) := C_{l} e^{-\frac{x^2}{2\sigma_n^2}} x^{l} Y_{lm}(x)
    # with
    #   C_l := 1
    assert normalize in ["none", "multipoles", "integrate", "receiver"]

    site = center
    x_m_xl = torch.clone(coordinates)
    x_m_xl[..., 0] -= site[0]
    x_m_xl[..., 1] -= site[1]
    x_m_xl[..., 2] -= site[2]

    x_m_xl_squared = torch.sum(torch.multiply(x_m_xl, x_m_xl), dim=-1)
    gauss_vals = torch.exp(-0.5 * x_m_xl_squared / sigma**2)
    r_tothe_l = torch.pow(x_m_xl_squared, l / 2)
    radial_part = torch.multiply(gauss_vals, r_tothe_l)  # [...,1]
    sp_vals = spherical_harmonics(
        l, permute_to_e3nn_convention(x_m_xl), normalize=True
    )  # [...,2l+1]

    basis_function = torch.multiply(sp_vals, radial_part.unsqueeze(-1))

    if normalize == "multipoles":
        l_dep_part = (
            (4 * np.pi / (2 * l + 1)) ** 0.5
            * 2 ** ((2 * l + 1) / 2)
            * scipy.special.gamma((2 * l + 3) / 2)
        )
        Cl_inverse = l_dep_part * sigma ** (2 * l + 3)  # [num_sigma, max_l+1]
    elif normalize == "integrate":
        Cl_inverse = (4 * np.pi / (2 * l + 1)) ** (-0.5)
    elif normalize == "receiver":
        Cl_inverse = (
            2 ** ((l + 1) / 2) * scipy.special.gamma((l + 3) / 2) * sigma ** (l + 3)
        )
    else:
        Cl_inverse = 1.0

    return basis_function / Cl_inverse  # [...,2l+1]


def tensor_realspace_Rlm_evaluation(coordinates, l, center):
    #   R_{lm}(x) := (4pi/(2l+1))**0.5 x^{l} Y_{lm}(x)
    site = center
    x_m_xl = torch.clone(coordinates)
    x_m_xl[..., 0] -= site[0]
    x_m_xl[..., 1] -= site[1]
    x_m_xl[..., 2] -= site[2]

    x_m_xl_squared = torch.sum(torch.multiply(x_m_xl, x_m_xl), dim=-1)
    radial_part = torch.pow(x_m_xl_squared, l / 2)  # [...,1]
    sp_vals = spherical_harmonics(
        l, permute_to_e3nn_convention(x_m_xl), normalize=True
    )  # [...,2l+1]

    prefactor = (4 * np.pi / (2 * l + 1)) ** 0.5
    basis_function = prefactor * torch.multiply(sp_vals, radial_part.unsqueeze(-1))

    return basis_function  # [...,2l+1]


def GTO_no_shift_two_site_overlap(
    coordinates, center1, center2, sigma1, sigma2, l1, l2
):
    # returns a (2*l1 + 1, 2*l2 + 1) matrix
    # where [i,j] is the overlap between m=i at center1 and m=j at center2.

    # evaluate both basis functions
    # these arrays are [*coords.shape, 2*l1+1] amnd [*coords.shape, 2*l1+1]
    phi1 = tensor_realspace_GTO_evaluation(coordinates, l1, sigma1, center1)
    phi2 = tensor_realspace_GTO_evaluation(coordinates, l2, sigma2, center2)

    square_integrals = torch.zeros((2 * l1 + 1, 2 * l2 + 1))

    for m1 in range(2 * l1 + 1):
        for m2 in range(2 * l2 + 1):
            values = torch.multiply(phi1[..., m1], phi2[..., m2])
            integ1 = torch.trapz(values, x=coordinates[..., 0], axis=0)
            integ2 = torch.trapz(integ1, x=coordinates[0, :, :, 1], axis=0)
            integ3 = torch.trapz(integ2, x=coordinates[0, 0, :, 2], axis=0)

            square_integrals[m1, m2] = integ3

    return square_integrals


def GTO_simple_shift_two_site_overlap(
    coordinates, cell, center1, center2, sigma1, sigma2, l1, l2
):
    # shift vectors
    shift_amounts = torch.cartesian_prod(
        torch.tensor([-1.0, 0.0, 1.0]),
        torch.tensor([-1.0, 0.0, 1.0]),
        torch.tensor([-1.0, 0.0, 1.0]),
    )
    shift_amounts = [torch.einsum("j,ji->i", shift, cell) for shift in shift_amounts]
    square_integrals = torch.zeros((2 * l1 + 1, 2 * l2 + 1))

    # all the coordinates are shifted by a cell vector
    for shifts in shift_amounts:
        thing1 = coordinates + shifts
        square_integrals += GTO_no_shift_two_site_overlap(
            coordinates + shifts, center1, center2, sigma1, sigma2, l1, l2
        )

    return square_integrals


def no_shifts_trapezoid_integral(values, coordinates):
    integ1 = torch.trapz(values, x=coordinates[..., 0], axis=0)
    integ2 = torch.trapz(integ1, x=coordinates[0, :, :, 1], axis=0)
    integ3 = torch.trapz(integ2, x=coordinates[0, 0, :, 2], axis=0)
    return integ3


def realspace_cluster_multipoles(coordinates, values, center, max_l):
    multipoles = torch.zeros(((max_l + 1) ** 2))

    for l in range(max_l + 1):
        Rl = tensor_realspace_Rlm_evaluation(coordinates, l, center)  # [...,2l+1]
        for m in range(2 * l + 1):
            product = torch.multiply(Rl[..., m], values)
            multipoles[l**2 + m] = no_shifts_trapezoid_integral(product, coordinates)

    return multipoles  # [ (max_Rlm_l+1)**2 ]
