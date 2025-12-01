################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
Not use in TACE now, but will update in future work,
There will be two completely different new implementations of irreducible Cartesian tensor products
It is expected to bring about a significant increase in speed and possible improvement in accuracy
'''

import torch
from torch import Tensor

from string import ascii_letters


LETTERS = list(ascii_letters)[3:]


def rotate_cart(T: Tensor, R: Tensor):
    r = T.ndim - 2 
    if r > 0:
        in_1 = 'b' + ''.join(LETTERS[0:2])
        in_2 = 'bc' + ''.join(LETTERS[2:r+2])
        in_2 = list(in_2)
        in_2[-1] = in_1[-1]
        in_2 = ''.join(in_2)
        out = 'bc' + in_1[1] + in_2[2:-1]
        einsum_str = in_1 + ',' + in_2 + '->' + out
        for _ in range(r):
            T = torch.einsum(einsum_str, R, T)
    return T


# def init_edge_rot_mat(edge_vector) -> torch.Tensor:
#     """
#     This function is adapted from eSCN: https://openreview.net/forum?id=QIejMwU0r9
#     Returns:
#         edge_rot_mat (edge, 3, 3).
#     """
#     edge_vec_0 = edge_vector
#     edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

#     norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

#     edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
#     edge_vec_2 = edge_vec_2 / (
#         torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
#     )
#     # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
#     # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
#     edge_vec_2b = edge_vec_2.clone()
#     edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
#     edge_vec_2b[:, 1] = edge_vec_2[:, 0]
#     edge_vec_2c = edge_vec_2.clone()
#     edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
#     edge_vec_2c[:, 2] = edge_vec_2[:, 1]
#     vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
#     vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

#     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
#     edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
#     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
#     edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

#     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
#     # Check the vectors aren't aligned
#     assert torch.max(vec_dot) < 0.99

#     norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
#     norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
#     norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
#     norm_y = torch.cross(norm_x, norm_z, dim=1)
#     norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

#     # Construct the 3D rotation matrix
#     norm_x = norm_x.view(-1, 3, 1)
#     norm_y = -norm_y.view(-1, 3, 1)
#     norm_z = norm_z.view(-1, 3, 1)

#     edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
#     edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

#     return edge_rot_mat.detach()
 

def init_edge_rot_mat(edge_vector) -> torch.Tensor:
    """
    x
    """
    edge_vec_0 = edge_vector
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1)) 

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / torch.sqrt(torch.sum(edge_vec_2**2, dim=1, keepdim=True))

    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]

    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1, keepdim=True))
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1, keepdim=True))
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1, keepdim=True))

    edge_vec_2 = torch.where(vec_dot > vec_dot_b, edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1, keepdim=True))
    edge_vec_2 = torch.where(vec_dot > vec_dot_c, edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    norm_y = torch.cross(norm_z, norm_x, dim=1)
    norm_y = norm_y / torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))

    norm_x = norm_x.view(-1, 3, 1)
    norm_y = norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_x, norm_y, norm_z], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()
