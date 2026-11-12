"""Exact sharing of equal radial functions on reciprocal homoelement edges.

Only connectivity, species and periodic shifts define the equivalence classes.
No coordinates, distances, learned activations or parameter values are cached.
Opposite vectors have the same norm as functions of positions and cell, so
sharing preserves forces, stress and all mixed parameter derivatives.
"""
import torch


@torch.no_grad()
def reciprocal_plan(batch, multiple=1):
    """Add representative indices and an inverse map for radial computations.

    Parameters
    ----------
    batch : dict
        Contains positions, atomic_numbers, edge_index and edge_shifts. Shifts
        must be the usual lattice-coordinate shifts used in the model graph.
    multiple : int
        Pad the representative capacity to this positive multiple. Extra
        representatives are unused by the inverse map.

    Returns
    -------
    dict
        Shallow copy with ``oam_radial_representatives`` and ``oam_radial_map``.
    """
    if multiple < 1:
        raise ValueError('The representative capacity multiple must be positive')
    source,target=batch['edge_index']
    shifts=batch['edge_shifts']
    numbers=batch['atomic_numbers']
    same=numbers[source]==numbers[target]
    # Positive first nonzero shift fixes the orientation of periodic self edges.
    negative=(shifts[:,0]<0)|((shifts[:,0]==0)&(shifts[:,1]<0))|((shifts[:,0]==0)&(shifts[:,1]==0)&(shifts[:,2]<0))
    reverse=same&((source>target)|((source==target)&negative))
    left=torch.where(reverse,target,source)
    right=torch.where(reverse,source,target)
    canonical_shifts=torch.where(reverse[:,None],-shifts,shifts)
    key=torch.cat((same[:,None].double(),left[:,None].double(),right[:,None].double(),canonical_shifts.double()),dim=1)
    unique,inverse=torch.unique(key,dim=0,sorted=True,return_inverse=True)
    count=unique.shape[0]
    representatives=source.new_full((count,),source.numel())
    representatives.scatter_reduce_(0,inverse,torch.arange(source.numel(),device=source.device),reduce='amin',include_self=True)
    padded=((count+multiple-1)//multiple)*multiple
    if padded>count:
        representatives=torch.cat((representatives,source.new_zeros((padded-count,))))
    result=dict(batch)
    result['oam_radial_representatives']=representatives
    result['oam_radial_map']=inverse
    return result
