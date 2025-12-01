################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO, This script is temporary, only used for developer, 
import argparse

import torch

from ..dataset.quantity import get_target_property, get_embedding_property
from ..lightning.select_model import select_model
from ..lightning.lit_model import LightningWrapperModel


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        # nargs='+',  
        help="Model ckpt path, try to remove direct_forces, direct_stress and direct_virials, other property are not supported now",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        # nargs='+',  
        choices=["dir2con", "stat"],
        help="Modify the parameters inside the model to transform it into a new one",
    )
    return parser.parse_args()


def load_subset_weights(conservative_model, direct_model):
    cons_state = conservative_model.state_dict()
    direct_state = direct_model.state_dict()

    cons_keys = set(cons_state.keys())
    direct_keys = set(direct_state.keys())

    if not cons_keys.issubset(direct_keys):
        missing = cons_keys - direct_keys
        raise ValueError(f"The following parameters are missing in direct_model: {missing}")

    for k in cons_keys:
        if cons_state[k].shape != direct_state[k].shape:
            raise ValueError(
                f"Shape mismatch for parameter '{k}': "
                f"conservative {cons_state[k].shape} vs direct {direct_state[k].shape}"
            )

    updated_state = {k: direct_state[k] for k in cons_keys}
    conservative_model.load_state_dict(updated_state, strict=True)

    return conservative_model

def direct_to_onservative_model(args):
    'Convert direct (energy forces stress virials) model) to conservative one'
    direct_model = LightningWrapperModel.load_from_checkpoint(
        args.model,
        map_location='cpu',
        weights_only=False,
        strict=True,
        use_ema=1,
    )
    checkpoint = torch.load(
        args.model, weights_only=False
    )
    cfg = checkpoint['hyper_parameters']['cfg']
    target_property = get_target_property(cfg)
    target_property_for_flags = target_property.copy()
    if 'direct_forces' in target_property: 
        target_property.remove('direct_forces')
    if 'direct_stress' in target_property: 
        target_property.remove('direct_stress')
    if 'direct_virials' in target_property: 
        target_property.remove('direct_virials')
    embedding_property = get_embedding_property(cfg)
    statistics = checkpoint['hyper_parameters']['statistics']
    conservative_model = select_model(
        cfg, statistics, target_property, embedding_property
    )
    conservative_model = load_subset_weights(conservative_model, direct_model)
    if 'direct_forces' in target_property_for_flags: 
        conservative_model.flags.compute_forces = True
        conservative_model.compute_first_derivative = True
    if 'direct_stress' in target_property_for_flags: 
        conservative_model.flags.compute_stress = True
        conservative_model.compute_first_derivative = True
    if 'direct_virials' in target_property_for_flags: 
        conservative_model.flags.compute_virials = True
        conservative_model.compute_first_derivative = True
    torch.save(conservative_model, "conservative_model.pt")
    print('Succesfully convert direct model to conservative model')

def main():
    args = parse_args()

    if args.type == 'dir2con':
        direct_model = LightningWrapperModel.load_from_checkpoint(
            args.model,
            map_location='cpu',
            weights_only=False,
            strict=True,
            use_ema=1,
        )
        checkpoint = torch.load(
            args.model, weights_only=False
        )
        cfg = checkpoint['hyper_parameters']['cfg']
        target_property = get_target_property(cfg)
        target_property_for_flags = target_property.copy()
        if 'direct_forces' in target_property: 
            target_property.remove('direct_forces')
        if 'direct_stress' in target_property: 
            target_property.remove('direct_stress')
        if 'direct_virials' in target_property: 
            target_property.remove('direct_virials')

        embedding_property = get_embedding_property(cfg)
        statistics = checkpoint['hyper_parameters']['statistics']
        conservative_model = select_model(
            cfg, statistics, target_property, embedding_property
        )
        conservative_model = load_subset_weights(conservative_model, direct_model)
        if 'direct_forces' in target_property_for_flags: 
            conservative_model.flags.compute_forces = True
            conservative_model.compute_first_derivative = True
        if 'direct_stress' in target_property_for_flags: 
            conservative_model.flags.compute_stress = True
            conservative_model.compute_first_derivative = True
        if 'direct_virials' in target_property_for_flags: 
            conservative_model.flags.compute_virials = True
            conservative_model.compute_first_derivative = True
        torch.save(conservative_model, "conservative_model.pt")
        print('Succesfully convert direct model to conservative model')
        return
    elif args.type == 'stat':
        model = LightningWrapperModel.load_from_checkpoint(
            args.model,
            map_location='cpu',
            weights_only=False,
            strict=True,
            use_ema=1,
        )
        with torch.no_grad():
            new_atomic_energies = [
                [
                    -1.1176, -0.0005, -0.2974, -0.0181, -0.4447, -1.3865, -3.1256, -1.9067, -0.7674, -0.0121, 
                    -0.2285, -0.0958, -0.3122, -0.8689, -1.8879, -1.0746, -0.3714, -0.0502, -0.2277, -0.0927, 
                    -2.2127, -2.6397, -3.7438, -5.6018, -5.3235, -3.5955, -2.1496, -1.0536, -0.6027, -0.1645, 
                    -0.4043, -0.8916, -1.6834, -0.8716, -0.2651, -0.0331, -0.1879, -0.068, -2.2868, -2.3603, 
                    -3.1513, -4.6011, -3.5438, -1.6595, -1.6479, -1.4776, -0.3388, -0.1672, -0.4087, -0.8167, 
                    -1.4107, -0.7239, -0.1703, -0.0097, -0.1369, -0.0344, -0.8455, -1.3876, -0.5491, -0.5186, 
                    -0.4895, -0.4683, -8.3662, -10.4088, -0.3982, -0.3886, -0.3834, -0.3857, -0.3168, -0.064, 
                    -0.3808, -3.527, -3.7421, -4.6555, -3.4276, -2.8979, -1.1789, -0.5638, -0.2872, -0.1235, 
                    -0.3606, -0.7674, -1.326, -0.3866, -1.1045, -2.553, -4.9889, -7.7017, -10.8084
                ]
            ]
            atomic_energy_list = []
            for atomic_energy in new_atomic_energies:
                atomic_energy_list.append([float(v) for v in atomic_energy])
            new_atomic_energy_tensor = torch.tensor(atomic_energy_list, dtype=torch.get_default_dtype())
            model.readout_fn.atomic_energy_layer.atomic_energy.copy_(new_atomic_energy_tensor)

            # new_scale = [
            #     [
            #         1.2479, 0.0284, 0.2406, 0.4126, 0.7826, 1.599, 1.5753, 0.8498, 0.6083, 0.0,
            #         0.1963, 0.1935, 0.4294, 0.7386, 1.2352, 0.6816, 0.4024, 0.0, 0.4444, 0.3384,
            #         0.3674, 0.5421, 0.8662, 0.6028, 0.3634, 0.3259, 0.373, 0.2487, 0.2266, 0.2833,
            #         0.6435, 0.3619, 0.6616, 1.0083, 0.3287, 0.2407, 0.331, 0.265, 0.2448, 0.8569,
            #         0.8551, 0.9837, 0.6988, 0.8013, 0.3093, 0.2268, 0.4571, 0.3263, 0.3278, 0.3482,
            #         0.5747, 0.6634, 0.4617, 0.7796, 0.1202, 0.2242, 0.341, 0.2357, 0.4187, 0.2967,
            #         0.2979, 0.3289, 0.1654, 0.1971, 0.2603, 0.3336, 0.2375, 0.2785, 0.3185, 0.4831,
            #         0.2021, 0.5809, 0.6663, 0.8977, 0.6913, 0.8224, 0.3624, 0.3336, 0.1913, 0.1713,
            #         0.6944, 0.1735, 0.4392, 0.0976, 0.345, 0.0914, 0.5779, 0.271, 0.3288
            #     ]
            # ]
            # new_scale_tensor = torch.tensor(new_scale, dtype=torch.get_default_dtype())
            # model.readout_fn.scale_shift.scale.copy_(new_scale_tensor)
        torch.save(model, "new_stat.pt")
   
  