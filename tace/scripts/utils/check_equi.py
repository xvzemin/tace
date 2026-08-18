# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################

# import argparse

# import ase.io
# import numpy as np
# import torch
# from ase import Atoms
# from e3nn import o3
# from torch_geometric.loader import DataLoader

# from tace.dataset.graph import from_atoms
# from tace.dataset.quantity import KeySpecification
# from tace.lightning import load_tace
# from tace.models.adapter import TensorModel


# def rotate_structure(atoms: Atoms, rotation: np.ndarray) -> Atoms:
#     rotated = atoms.copy()
#     rotated.positions[:] = atoms.positions @ rotation.T
#     rotated.set_cell(np.asarray(atoms.cell) @ rotation.T, scale_atoms=False)
#     return rotated

# def invert_structure(atoms: Atoms) -> Atoms:
#     inverted = atoms.copy()
#     inverted.positions[:] = -atoms.positions
#     inverted.set_cell(-np.asarray(atoms.cell), scale_atoms=False)
#     return inverted

# def build_graph(atoms: Atoms, model: TensorModel):
#     max_neighbors = model.get_max_neighbors()
#     return from_atoms(
#         model.get_torch_element(),
#         atoms,
#         model.get_cutoff(),
#         max_neighbors="inf" if max_neighbors is None else max_neighbors,
#         keyspec=KeySpecification.from_defaults(),
#         target_property=[],
#         embedding_property=model.get_embedding_property(),
#         training=False,
#         neighborlist_backend="matscipy",
#     )


# def predict_descriptor(
#     atoms: Atoms,
#     model: TensorModel,
#     device: torch.device,
# ) -> np.ndarray:
#     graph = build_graph(atoms, model)
#     num_channel = model.readout_fn.model_config["num_channel"]
#     loader = DataLoader([graph], batch_size=1, shuffle=False)
#     batch = next(iter(loader))
#     output = model(batch.to(device))
#     # descriptor = output["descriptors"][-1][:, num_channel:num_channel*4].reshape(-1, num_channel, 3)
#     descriptor = output["descriptors"][-1][:, num_channel*4:num_channel*7].reshape(-1, num_channel, 3)
#     return descriptor.detach().cpu().numpy()


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-m",
#         "--model",
#         required=True,
#         help="TACE model.",
#     )
#     parser.add_argument(
#         "-i",
#         "--input",
#         required=True,
#         help="ASE-readable structures.",
#     )

#     return parser.parse_args()


# def main():
#     args = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_tace(
#         args.model,
#         device=device,
#         dtype="float64",
#     )
#     model.eval()
#     atoms_list = ase.io.read(args.input, index=":1")
#     torch.manual_seed(42)

#     # rotations = o3.rand_matrix(3, dtype=torch.float64).cpu().numpy()
#     # errors = []
#     # for atoms in atoms_list:
#     #     descriptor = predict_descriptor(atoms, model, device)
#     #     for rotation in rotations:
#     #         rotated_atoms = rotate_structure(atoms, rotation)
#     #         rotated_descriptor = predict_descriptor(rotated_atoms, model, device)
#     #         expected_descriptor = descriptor @ rotation.T
#     #         errors.append((rotated_descriptor - expected_descriptor).reshape(-1))
#     # errors = np.concatenate(errors)
#     # mae = np.mean(np.abs(errors))
#     # rmse = np.sqrt(np.mean(errors**2))
#     # print(f"Rotation MAE:  {mae:.8e}")
#     # print(f"Rotation RMSE: {rmse:.8e}")

#     # errors = [] # 1o
#     # for atoms in atoms_list:
#     #     descriptor = predict_descriptor(atoms, model, device)
#     #     expected_descriptor = -1 * descriptor
#     #     inversion_atoms = invert_structure(atoms)
#     #     inversion_descriptor = predict_descriptor(inversion_atoms, model, device)
#     #     errors.append((inversion_descriptor - expected_descriptor).reshape(-1))

#     errors = [] # 1e
#     for atoms in atoms_list:
#         descriptor = predict_descriptor(atoms, model, device)
#         expected_descriptor = 1 * descriptor
#         inversion_atoms = invert_structure(atoms)
#         inversion_descriptor = predict_descriptor(inversion_atoms, model, device)
#         errors.append((inversion_descriptor - expected_descriptor).reshape(-1))

#     errors = np.concatenate(errors)
#     mae = np.mean(np.abs(errors))
#     rmse = np.sqrt(np.mean(errors**2))

#     print(f"Inversion MAE:  {mae:.8e}")
#     print(f"Inversion RMSE: {rmse:.8e}")

# if __name__ == "__main__":
#     main()