################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import warnings
from typing import Optional, List


import torch
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes
from torch_geometric.loader import DataLoader

from ...dataset.quantity import PROPERTY
from ...dataset.element import TorchElement
from ...dataset.graph import from_atoms
from ...lightning.lit_model import LightningWrapperModel
from ...dataset.quantity import (
    PROPERTY,
    KEYS,
    KeySpecification,
    update_keyspec_from_kwargs,
)


class TACECalculator(Calculator):
    """
    Initialize a TACECalculator. We support the most fundamental potential energy surface property and multi-fidelity, 
    multi-head, etc. For some advanced features, you need to modify the code yourself and store the attributes that need 
    to be embedded in atoms.info or atmoity.arrays. If you only need to predict, you can directly use the `tace-eval` 
    command. It will output the predicted files, and if you add the `--test` option, it will also output the errors.

    Parameters
    ----------
    model_path : str
        Path to the trained model, file ends with ``pt, .pth or .ckpt``.
    device : str, default='cpu'
        The device to run computations on, e.g., ``cpu`` or ``cuda``.
    dtype : str, optional, default=None
        Data type for computations, e.g., ``float32`` or ``float64``.
    max_neighbors : int, optional, default=None
        Maximum number of neighbors to consider for each atom. Can be an ``positive integer`` or ``None``.
        ``None`` means use the same setting in training.
    extra_compute_first_derivative : list[str], optional, default=None
        If you wand to predict property not trained in your model, 
        You need to provide the names of the first-order derivative physical quantities for additional predictions.
        For example, if model trained on energy only, you colud also predict forces, and stress.
    extra_compute_second_derivative : list[str], optional, default=None
        If you wand to predict property not trained in your model, 
        You need to provide the names of the second-order derivative physical quantities for additional predictions.
        For example, if model trained on energy forces only, you colud also predict hessians.
        One another example is that if model trained on conservative_dipole under electric_field, you colud also predict 
        conservative_polarizability.
    level : int
        Specify which fidelity level to use. The default is a single head,
        i.e., the index of fidelity `level` defaults to 0.
    **kwargs
        Additional keyword arguments passed to the ASE Calculator base class.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        max_neighbors: Optional[int] = None,
        extra_compute_first_derivative: Optional[List[str]] = None,
        extra_compute_second_derivative: Optional[List[str]] = None,
        level: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.level = level 
        self.extra_compute_first_derivative = extra_compute_first_derivative or []
        self.extra_compute_second_derivative = extra_compute_second_derivative or []

        if model_path.endswith(".ckpt"):
            model = LightningWrapperModel.load_from_checkpoint(
                model_path,
                map_location=device,
                strict=True,
                use_ema=True,
            )
        elif model_path.endswith(".pt") or model_path.endswith(".pth"):
            model = torch.load(model_path, weights_only=False, map_location=device)
        else:
            raise ValueError("❌ Model path must end with '.ckpt', '.pt', or '.pth'")
        
        if dtype is not None:
            model_dtype = model.readout_fn.cutoff.dtype
            if dtype == "float64":
                torch.set_default_dtype(torch.float64)
                model = model.double()
                if model_dtype != torch.float64:
                    warnings.warn(
                        f"Model dtype {model_dtype} != default dtype {dtype}. "
                        f"This may cause silent type conversions."
                    )
            elif dtype == "float32":
                torch.set_default_dtype(torch.float32)
                model = model.float()
                if model_dtype != torch.float32:
                    warnings.warn(
                        f"Model dtype {model_dtype} != default dtype {dtype}. "
                        f"This may cause silent type conversions."
                    )
            else:
                raise ValueError(f"Unknown dtype {dtype}")
        else:
            torch.set_default_dtype(model_dtype)
            
        target_property = model.target_property
        compute_flags = {}
        for p in self.extra_compute_first_derivative:
            model.compute_first_derivative = True
            compute_flags.update(
                {
                    p: True
                }
            )
        for p in self.extra_compute_second_derivative:
            model.compute_second_derivative = True
            compute_flags.update(
                {
                    p: True
                }
            )
        for p, flag in compute_flags.items():
            if flag:
                setattr(model.flags, f"compute_{p}", True)
                target_property.append(p)
        self.target_property = list(set(target_property))
        self.embedding_property = model.embedding_property
        self.implemented_properties = self.target_property + ["free_energy"]
        self.universal_embedding = model.universal_embedding
        self.max_neighbors = getattr(model.readout_fn, "max_neighbors", max_neighbors)
        self.cutoff = float(model.readout_fn.cutoff.item())
        self.element = TorchElement([int(z) for z in model.readout_fn.atomic_numbers.cpu().tolist()])

        for param in model.parameters():
            param.requires_grad = False

        self.keySpecification = KeySpecification()
        update_keyspec_from_kwargs(self.keySpecification, KEYS)

        self.device = device
        self.model = model.to(device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        atoms.info['level'] = self.level # fidelity level
        # === dataloader ===
        data = [
            from_atoms(
                self.element,
                atoms,
                self.cutoff,
                max_neighbors=self.max_neighbors,
                target_property=self.target_property,
                embedding_property=self.embedding_property,
                keyspec=self.keySpecification,
                universal_embedding=self.universal_embedding,
                training=False,
            ) 
        ]
        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch = next(iter(dataloader)).to(self.device)
        for p in self.target_property:
            for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                batch[requires_grad_p].requires_grad_(True)

        # === forward ===
        outs = self.model(batch)
        # === update ===
        self.results = {}
        for p in self.target_property:
            p_type = PROPERTY[p]['type']
            p_rank = PROPERTY[p]['rank']
            prop = outs[p]
            if p_type == 'graph':
                if p_rank == 0:
                    if p == 'energy':
                        energy = prop.detach().cpu().item()
                        self.results[p] = energy
                        self.results["free_energy"] = self.results[p]
                    else:
                        self.results[p] = prop.detach().cpu().item()
                elif set([p]) & {'stress', 'virials'} :
                    prop = prop.detach().cpu().numpy().squeeze(0)
                    prop = full_3x3_to_voigt_6_stress(prop)
                    self.results[p] = prop
                else:
                    self.results[p] = outs[p].detach().cpu().numpy().squeeze(0)
            elif p_type == 'atom':
                self.results[p] = prop.detach().cpu().numpy()
            else:
                raise

            
    def get_hessians(self, atoms=None):
        self.target_property = list(set(self.target_property+'hessians'))
        self.model.compute_forces = True
        self.model.compute_hessians = True
        self.model.compute_first_derivative = True

        data = [
            from_atoms(
                self.element,
                atoms,
                self.cutoff,
                max_neighbors=self.max_neighbors,
                target_property=self.target_property,
                embedding_property=self.embedding_property,
                keyspec=self.keySpecification,
                universal_embedding=self.universal_embedding,
                training=False,
            ) 
        ]

        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch = next(iter(dataloader))
        batch.to(self.device)
        for p in self.target_property:
            for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                batch[requires_grad_p].requires_grad_(True)
        outs = self.model(batch)

        return outs["hessians"].detach().cpu().numpy() 


    def get_direct_polarizability(self, atoms=None):
        self.target_property = list(set(self.target_property+'direct_polarizability'))
        data = [
            from_atoms(
                self.element,
                atoms,
                self.cutoff,
                max_neighbors=self.max_neighbors,
                target_property=self.target_property,
                embedding_property=self.embedding_property,
                keyspec=self.keySpecification,
                universal_embedding=self.universal_embedding,
                training=False,
            ) 
        ]

        dataloader = DataLoader(
            dataset=data,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch = next(iter(dataloader))
        batch.to(self.device)
        for p in self.target_property:
            for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                batch[requires_grad_p].requires_grad_(True)
        outs = self.model(batch)
        return outs['direct_polarizability'] .detach().cpu().numpy() 
