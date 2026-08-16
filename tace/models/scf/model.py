################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict

from tace.models._e3nn.tace import e3nnTACE

from .energy_functional import EnergyFunctionalSCF
from .fixed_point import FixedPointSCF
from .local import FixedChargeBaseline, LocalCharges, LocalSplitCharges
from .qeq import QEq


class SCFTACE(e3nnTACE):
    """TACE backbone with a configurable electrostatic or SCF model family."""

    def __init__(self, scf: Dict, **kwargs) -> None:
        if not isinstance(scf, dict):
            raise TypeError("scf must be a dictionary")
        original_scf = dict(scf)
        scf = dict(scf)
        super().__init__(**kwargs)
        self.model_config["scf"] = original_scf
        self.model_config["_target_"] = "tace.models.scf.SCFTACE"

        method = scf.pop("method", None)
        if method is None:
            raise ValueError("scf.method must be provided")
        irreps_in = [product.irreps_out for product in self.representation.products]
        common = {"irreps_in": irreps_in}
        scalar_descriptor_dim = self.num_channel * self.num_layers

        if method in {"local_charge", "local_charges"}:
            self.scf_method = LocalCharges(**common, **scf)
        elif method in {"local_split_charge", "local_split_charges"}:
            self.scf_method = LocalSplitCharges(
                **common,
                scalar_descriptor_dim=scalar_descriptor_dim,
                num_elements=self.num_elements,
                **scf,
            )
        elif method in {"fixed_charge", "fixed_charge_baseline"}:
            self.scf_method = FixedChargeBaseline(num_elements=self.num_elements, **scf)
        elif method == "qeq":
            self.scf_method = QEq(**common, **scf)
        elif method == "fixed_point":
            self.scf_method = FixedPointSCF(**common, **scf)
        elif method == "energy_functional":
            self.scf_method = EnergyFunctionalSCF(**common, **scf)
        else:
            raise ValueError(
                "scf.method must be local_charges, local_split_charges, "
                "fixed_charge_baseline, qeq, fixed_point, or energy_functional"
            )
        self.scf_method_name = method
        required_embedding = []
        if method in {"qeq", "fixed_point", "energy_functional"}:
            required_embedding.append("total_charge")
        if method in {
            "local_split_charge",
            "local_split_charges",
            "fixed_charge",
            "fixed_charge_baseline",
        } and scf.get("formal_charges_from_data", True):
            required_embedding.append("charges")
        self.embedding_property = list(
            dict.fromkeys([*self.embedding_property, *required_embedding])
        )

    def forward(self, data, graph):
        base_results = super().forward(data, graph)
        return self.scf_method(data, graph, base_results)
