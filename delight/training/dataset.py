import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore
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
        self.X_raw = np.load(os.path.join(options.source, X_path))
        self.y = self.transform(
            np.load(os.path.join(options.source, y_path)),
            options.rot,
            options.flip,
        )
        self.y_raw = np.load(os.path.join(options.source, y_path))
        

    @staticmethod
    def transform(
        y: np.ndarray[Any, np.dtype[np.float32]], rot: bool, flip: bool
    ) -> torch.Tensor:
        transformed: tuple[np.ndarray[Any, np.dtype[np.float32]], ...]

        if rot is False and flip is False:
            return torch.Tensor(y)

        yflip = cast(np.ndarray[Any, np.dtype[np.float32]], [1, -1] * y)
        if rot is False:
            transformed = (y, yflip)

        y90 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y[:, ::-1])
        y180 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y90[:, ::-1])
        y270 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y180[:, ::-1])
        yflip90 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * yflip[:, ::-1])
        yflip180 = cast(
            np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * yflip90[:, ::-1]
        )
        yflip270 = cast(
            np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * yflip180[:, ::-1]
        )

        if flip is False:
            transformed = (y, y90, y180, y270)
        else:
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

    def to_tf_dataset(self) -> tuple[tf.Tensor, tf.Tensor]:
        X = cast(np.ndarray[Any, np.dtype[np.float32]], self.X.numpy())
        y = cast(np.ndarray[Any, np.dtype[np.float32]], self.y.numpy())

        return tnp.copy(X.transpose((0, 2, 3, 1))), tnp.copy(y)  # type: ignore
