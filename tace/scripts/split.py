################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import random
import argparse


from ase import Atoms
from ase.io import read, write


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val/test sets with info['xzm_index'] labeling"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the structure file, e.g. data.traj or data.xyz",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        required=True,
        help="Three non-negative integers: number of train, val, and test samples",
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_index", type=str, default=None, help="Split from index file (index start from zero)")
    parser.add_argument("--valid_index", type=str, default=None, help="Split from index file (index start from zero)")
    parser.add_argument("--test_index", type=str, default=None, help="Split from index file (index start from zero)")
    return parser.parse_args()


def main():

    args = parse_args()
    structures: list[Atoms] = read(args.input, index=":")
    total = len(structures)
    print(f"Total structures loaded: {total}")

    # Requested split sizes
    n_train, n_val, n_test = args.num
    total_requested = n_train + n_val + n_test

    if total_requested == 0:
        print("Nothing to split: all target sizes are 0.")
        return

    if total_requested > total:
        raise ValueError(
            f"Requested total of {total_requested} structures "
            f"(train={n_train}, val={n_val}, test={n_test}), "
            f"but only {total} structures are available in the file."
        )

    if total_requested == total:
        print(f"All structures in {args.input} have benn used.")

    print(f"Final split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Shuffle and assign indices
    random.seed(args.seed)
    indices = list(range(total))
    random.shuffle(indices)

    split_tags = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    tag_dict = {"train": [], "val": [], "test": []}

    for i, idx in enumerate(indices[: len(split_tags)]):
        tag = split_tags[i]
        atoms = structures[idx]
        atoms.info["tace_index"] = idx  # <-- original index
        tag_dict[tag].append(atoms)

    base_name = args.input.rsplit(".", 1)[0]

    for tag, atoms_list in tag_dict.items():
        if atoms_list:
            out_file = f"{base_name}_{tag}.xyz"
            write(out_file, atoms_list)
            print(f"Wrote {len(atoms_list)} structures to {out_file}")


if __name__ == "__main__":
    main()
