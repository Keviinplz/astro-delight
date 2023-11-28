import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class DelightDatasetType(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VALIDATION = "VALIDATION"


@dataclass
class DelightDatasetOptions:
    source: str
    n_levels: int
    fold: int
    mask: bool
    object: bool
    rot: bool
    flip: bool

    def get_filenames(self, datatype: DelightDatasetType) -> tuple[str, str]:
        if datatype == DelightDatasetType.TRAIN:
            x = "X_train_nlevels%i_fold%i_mask%s_objects%s.npy" % (
                self.n_levels,
                self.fold,
                self.mask,
                self.object,
            )
            y = "y_train_nlevels%i_fold%i_mask%s_objects%s.npy" % (
                self.n_levels,
                self.fold,
                self.mask,
                self.object,
            )
        elif datatype == DelightDatasetType.TEST:
            x = "X_test_nlevels%i_mask%s_objects%s.npy" % (
                self.n_levels,
                self.mask,
                self.object,
            )
            y = "y_test_nlevels%i_mask%s_objects%s.npy" % (
                self.n_levels,
                self.mask,
                self.object,
            )
        else:
            x = "X_val_nlevels%i_fold%i_mask%s_objects%s.npy" % (
                self.n_levels,
                self.fold,
                self.mask,
                self.object,
            )
            y = "y_val_nlevels%i_fold%i_mask%s_objects%s.npy" % (
                self.n_levels,
                self.fold,
                self.mask,
                self.object,
            )

        return x, y


class DelightDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, options: DelightDatasetOptions, datatype: DelightDatasetType):
        X_path, y_path = options.get_filenames(datatype)
        self.X = torch.Tensor(np.load(os.path.join(options.source, X_path))).permute(
            0, 3, 1, 2
        )
        self.y = self.transform(
            np.load(os.path.join(options.source, y_path)),
            options.rot,
            options.flip,
        )

    @staticmethod
    def transform(
        y: np.ndarray[Any, np.dtype[np.float32]], rot: bool, flip: bool
    ) -> torch.Tensor:
        if rot is False and flip is False:
            return torch.Tensor(y)

        if rot is False:
            yflip = [1, -1] * y
            transformed = (y, yflip)

        elif flip is False:
            y90 = [-1, 1] * y[:, ::-1]
            y180 = [-1, 1] * y90[:, ::-1]
            y270 = [-1, 1] * y180[:, ::-1]
            transformed = (y, y90, y180, y270)

        else:
            y90 = [-1, 1] * y[:, ::-1]
            y180 = [-1, 1] * y90[:, ::-1]
            y270 = [-1, 1] * y180[:, ::-1]
            yflip = [1, -1] * y
            yflip90 = [-1, 1] * yflip[:, ::-1]
            yflip180 = [-1, 1] * yflip90[:, ::-1]
            yflip270 = [-1, 1] * yflip180[:, ::-1]
            transformed = (y, y90, y180, y270, yflip, yflip90, yflip180, yflip270)

        return torch.Tensor(np.concatenate(transformed, axis=1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]

        if len(x.shape) == 3:  # has no channel information
            levels, width, height = x.shape
            x = x.reshape(levels, 1, width, height)  # asume 1 channel information
        return x, y