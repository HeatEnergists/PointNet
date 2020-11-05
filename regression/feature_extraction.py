import os
from concurrent import futures

import numpy as np
import pandas as pd

FEATURE_DIMMENSION = 29
FILE_NUMBER = 133886
xyz_file_folder = "coordinate//coordinate"

atom_parameters = pd.DataFrame(
    [
        [12.01, 6, 4, 2.55, 1.70, 0.75],
        [1.008, 1, 1, 2.20, 1.10, 0.32],
        [16.00, 8, 6, 3.44, 1.52, 0.64],
        [14.01, 7, 5, 3.04, 1.55, 0.71],
        [19.00, 9, 7, 3.98, 1.47, 0.60],
    ]
)
atom_parameters.index = ["C", "H", "O", "N", "F"]
atom_parameters.columns = ["mass", "electron", "outer_electron", "a", "b", "c"]


def generate_feture(aera):
    feature_dataset = []
    target_dataset = []
    for k in aera:
        with open(
            os.path.join(xyz_file_folder, "dsgdb9nsd_" + "{:0>6}.xyz".format(k)), "r"
        ) as file:
            results = file.readlines()
            target = np.array([float(results[1].split()[-3])],dtype=np.float32)
            num_of_atoms = int(results[0])
            useful_lines = results[2 : num_of_atoms + 2]
            atom_information = pd.DataFrame(
                [atom.replace("*^", "e").split()[:-1] for atom in useful_lines]
            )
            atom_information.columns = ["atom", "x", "y", "z"]
            atom_information.loc[:, "x":"z"] = atom_information.loc[:, "x":"z"].astype(
                float
            )
            atom_information.loc[:, "mass"] = [
                atom_parameters.loc[atom].mass for atom in atom_information.atom
            ]
            for scaled_coordinate, original_coordinate in zip(
                ["scaled_x", "scaled_y", "scaled_z"], ["x", "y", "z"]
            ):
                center = (
                    np.sum(
                        atom_information.loc[:, original_coordinate]
                        * atom_information.loc[:, "mass"]
                    )
                    / atom_information.loc[:, "mass"].sum()
                )
                atom_information.loc[:, scaled_coordinate] = (
                    atom_information.loc[:, original_coordinate] - center
                )
            input_array = np.concatenate(
                [
                    atom_information.loc[
                        :, ["scaled_x", "scaled_y", "scaled_z", "mass"]
                    ].values,
                    np.zeros((FEATURE_DIMMENSION - num_of_atoms, 4)),
                ]
            )
            input_array = np.array(input_array,dtype=np.float32)
            feature_dataset.append(input_array)
            target_dataset.append(target)
    return [feature_dataset, target_dataset]


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    with futures.ProcessPoolExecutor(8) as executor:
        datasets = list(
            executor.map(generate_feture, np.array_split(range(1, FILE_NUMBER), 8))
        )
    all_feature = np.concatenate([dataset[0] for dataset in datasets])
    all_target = np.concatenate([dataset[1] for dataset in datasets])
    np.save("feature.npy", all_feature)
    np.save("target.npy", all_target)
