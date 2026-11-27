"""TACE block selection for eager and rank-local execution."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

from tace.models.cartesian.prod import NonlinearIctdACE
from tace.models.cartesian.tace import CartesianTACE
from tace.models.spherical.inter import (
    O2CgtpInteraction,
    O3CgtpInteraction,
    UuO2Interaction,
    UvO2Interaction,
)
from tace.models.spherical.mtace import (
    O2MagneticInteraction,
    O3MagneticWigner6jInteraction,
)
from tace.models.spherical.prod import CgtpACE, MaceACE, NonlinearCgtpACE, TaceACE
from tace.models.spherical.tece_oam_rra import BilinearMoEACE, UvSO2Interaction
from tace.utils.env import memory_first

from .matching import custom_forward, custom_message_call, native_spherical_message


@dataclass(frozen=True)
class BlockRule:
    """Host layer types, structural matcher and supported execution scope."""

    types: tuple[type, ...]
    select: Callable
    rank_local: bool = False
    provider: Callable | None = None


def retain(module, *, backend, task):
    return None, "no eligible block rule for this execution scope"


def select_block(module, rules, *, rank_local, spherical_messages):
    for rule in rules:
        if isinstance(module, rule.types):
            # Single-device lowering can retarget an already scheduled subclass.
            # MPI accepts only reviewed native types and the original call graph.
            installed = (
                getattr(module, "_foundation_regional_provider", None) is not None
            )
            if (
                (rank_local and not rule.rank_local)
                or ((rank_local or not installed) and type(module) not in rule.types)
                or custom_forward(module, recursive=True)
                or (hasattr(module, "rejector") and custom_message_call(module))
                or (
                    spherical_messages
                    and not installed
                    and not native_spherical_message(module)
                )
            ):
                return retain, None
            return rule.select, rule.provider
    return retain, None


def block_rules(model, *, backend, task, automatic=False, rank_local=False):
    """Return interaction and product matchers for a loaded model and task."""
    from .products import host_product_provider, product_rule

    def bind(*rules, spherical_messages=False):
        return partial(
            select_block,
            rules=rules,
            rank_local=rank_local,
            spherical_messages=spherical_messages,
        )

    if isinstance(model.readout_fn, CartesianTACE):
        from .cartesian import (
            cartesian_interaction_rule,
            cartesian_product_rule,
            paired_cartesian_stack,
        )

        rep = model.readout_fn.representation
        measured_stack = (
            automatic
            and not rank_local
            and type(model.readout_fn) is CartesianTACE
            and len(rep.products) == len(rep.interactions) == 2
            and all(type(p) is NonlinearIctdACE for p in rep.products)
            and all(
                type(m) is O3CgtpInteraction and (m.Lmax, m.lmax, m.mmax) == (2, 3, 3)
                for m in rep.interactions
            )
        )
        return (
            bind(
                BlockRule(
                    (UvO2Interaction,),
                    partial(cartesian_interaction_rule, automatic=automatic),
                ),
                BlockRule(
                    (UuO2Interaction, O3CgtpInteraction),
                    partial(
                        cartesian_interaction_rule,
                        paired_radial=(
                            not automatic
                            and not rank_local
                            and backend == "cuda"
                            and not torch.version.hip
                            and paired_cartesian_stack(model)
                        ),
                    ),
                    rank_local=True,
                ),
            ),
            bind(
                BlockRule(
                    (BilinearMoEACE,),
                    partial(product_rule, automatic=automatic),
                    rank_local=True,
                ),
                BlockRule(
                    (NonlinearIctdACE, CgtpACE),
                    partial(
                        cartesian_product_rule,
                        automatic=automatic,
                        measured_stack=measured_stack,
                    ),
                    rank_local=True,
                ),
            ),
        )

    from tace.models.physical_basis import (
        ExpWeightedjnSphericalBesselBasis,
        SpacePhysicalBasis,
    )

    from .interactions import (
        canonical_geometry,
        grouped_radial_geometry,
        host_interaction_provider,
        interaction_rule,
        rotary_radial_geometry,
    )
    from .local import (
        aligned_rule,
        host_aligned_provider,
        local_geometry,
        local_rule,
        measured_aligned_stack,
    )
    from .magnetic import magnetic_rule
    from .tece_oam_rra.regional import rotary_rule
    from .wigner6j import host_wigner6j_provider, wigner6j_rule

    rep = model.readout_fn.representation
    canonical = not rank_local and canonical_geometry(rep)
    cuda = backend == "cuda" and not automatic and not torch.version.hip
    parameter = next(model.parameters(), None)
    cgtp = partial(
        interaction_rule,
        canonical=canonical,
        grouped_radial=(
            canonical
            and cuda
            and getattr(parameter, "dtype", None) == torch.float32
            and memory_first()
            and grouped_radial_geometry(rep)
            and (
                task == "training"
                or type(rep.radial_basis.radial_fn)
                in (ExpWeightedjnSphericalBesselBasis, SpacePhysicalBasis)
            )
        ),
        shared_radial=cuda
        and (rank_local or (memory_first() and rotary_radial_geometry(rep))),
    )
    local = partial(local_rule, canonical=not rank_local and local_geometry(rep))
    aligned = partial(
        aligned_rule,
        automatic=automatic,
        measured_stack=automatic
        and not rank_local
        and backend == "cuda"
        and measured_aligned_stack(model),
    )
    # More specific host types precede their base classes.
    return (
        bind(
            BlockRule(
                (O3MagneticWigner6jInteraction,),
                wigner6j_rule,
                provider=host_wigner6j_provider,
            ),
            BlockRule(
                (O2MagneticInteraction,),
                partial(magnetic_rule, automatic=automatic),
                provider=host_interaction_provider,
            ),
            BlockRule(
                (UvSO2Interaction,),
                partial(rotary_rule, automatic=automatic),
                rank_local=True,
                provider=host_interaction_provider,
            ),
            BlockRule(
                (UuO2Interaction, UvO2Interaction),
                local,
                rank_local=True,
                provider=host_interaction_provider,
            ),
            BlockRule(
                (O2CgtpInteraction,),
                aligned,
                provider=host_aligned_provider,
            ),
            BlockRule(
                (O3CgtpInteraction,),
                cgtp,
                rank_local=True,
                provider=host_interaction_provider,
            ),
            spherical_messages=True,
        ),
        bind(
            BlockRule(
                (CgtpACE, NonlinearCgtpACE, MaceACE, TaceACE, BilinearMoEACE),
                partial(product_rule, automatic=automatic),
                rank_local=True,
                provider=host_product_provider,
            ),
        ),
    )
