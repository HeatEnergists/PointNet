import math

import numpy as np
import torch.utils.data as Data


class MolecularDataset(Data.Dataset):
    def __init__(
        self,
        feature_file="feature.npy",
        target_file="target.npy",
        transform=None,
        feature_dimension=29,
        data_augmentation=True,
    ):

        self.features = np.load(feature_file, allow_pickle=True)
        self.targets = np.load(target_file, allow_pickle=True)
        self.transform = transform
        self.feature_dimension = feature_dimension
        self.data_augmentation = data_augmentation

    def _preprocess(self, feature):
        if self.data_augmentation:
            # rotate along x axis
            theta_x = np.random.uniform(0, np.pi * 2)
            rotation_matrix_x = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(theta_x), -math.sin(theta_x)],
                    [0, math.sin(theta_x), math.cos(theta_x)],
                ]
            )
            feature[:, :3] = feature[:, :3] @ (rotation_matrix_x)

            # rotate along y axis
            theta_y = np.random.uniform(0, np.pi * 2)
            rotation_matrix_y = np.array(
                [
                    [math.cos(theta_y), 0, math.sin(theta_y)],
                    [0, 1, 0],
                    [-math.sin(theta_y), 0, math.cos(theta_y)],
                ]
            )
            feature[:, :3] = feature[:, :3] @ (rotation_matrix_y)

            # rotate along z axis
            theta_z = np.random.uniform(0, np.pi * 2)
            rotation_matrix_z = np.array(
                [
                    [math.cos(theta_z), -math.sin(theta_z), 0],
                    [math.sin(theta_z), math.cos(theta_z), 0],
                    [0, 0, 1],
                ]
            )
            feature[:, :3] = feature[:, :3] @ (rotation_matrix_z)

            # random jitter
            feature[:, :3] = feature[:, :3] + np.random.normal(
                0, 0.01, size=(self.feature_dimension, 3)
            )

        if self.transform is not None:
            feature = self.transform(feature)

    def __getitem__(self, index):
        feature = self.features[index]
        target = self.targets[index]
        self._preprocess(feature)
        return feature, target

    def __len__(self):

        return self.features.shape[0]
