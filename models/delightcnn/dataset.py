import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore
import torch
from torch.utils.data import Dataset


class DelightDatasetType(Enum):
    """
    Defines which type of dataset it will use.

    TRAIN: Raw train dataset of DELIGHT, read from `.npy` file. No balance strategy is used.
    TEST: Raw test dataset of DELIGHT, read from `.npy` file.
    VALIDATION: Raw validation dataset of DELIGHT, read from `.npy` file.
                Data is masked where mean of pixeles (distance * pixscale) has to be less than 60.
    P_TRAIN: Concatenation of `train` and `validation` set. Used when model is training to production.
    P_VAL: Just the `test` data. Used when model is training to production.
    """

    TRAIN = "TRAIN"
    TEST = "TEST"
    VALIDATION = "VALIDATION"
    P_TRAIN = "PRODUCTION_TRAIN"
    P_VAL = "PRODUCTION_VALIDATION"


@dataclass
class DelightDatasetOptions:
    """
    Defines where to find data and how to transform it.

    Attributes:
    - source: Absolute path of where `.npy` files are located.
    - n_levels: Selects data with `n_levels` levels.
    - fold: Selects data from `fold` fold.
    - object: Flag that choose data with objects removed or not.
    - rot: Flag that indicates if a rotation transformation has to be used on the data to generate a data augmentation.
    - flip: Flag that indicates if a flip transformation has to be used on the data to generate a data augmentation.
    - balance: Flat that uses a strategy to balance data in `TRAIN`. Defaults to `True`.
    """

    source: str
    n_levels: int
    fold: int
    mask: bool  # TODO: sacar
    object: bool
    rot: bool
    flip: bool
    balance: bool = True


class DelightDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        options: DelightDatasetOptions,
        datatype: DelightDatasetType,
        transform_y: bool = True,
    ):
        """
        Initialize an instance of a `torch.utils.data.Dataset` object.

        Attributes:
        - options: A object that defines data location and transformations to use.
        - datatype: A enum that defines which kind of data to use.
        - transform_y: Flag that indicates if labels will be transformed using information from `options`. Defaults to `True`.
        """
        X, y = self.get_data(options, datatype)

        self.X = torch.Tensor(X).permute(0, 3, 1, 2)

        self.y = (
            self.transform(
                y,
                options.rot,
                options.flip,
            )
            if transform_y
            else torch.from_numpy(y)  # type: ignore
        )

    @classmethod
    def get_data(
        cls, options: DelightDatasetOptions, datatype: DelightDatasetType
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Given an option and a data type, returns a raw dataset as tuple of observations and labels.

        Attributes:
        - options: A object that defines data location and transformations to use.
        - datatype: A enum that defines which kind of data to use.

        Returns:
        - Tuple of two numpy arrays (X, y)
        """
        enum = {
            DelightDatasetType.TRAIN: cls.get_train_data,
            DelightDatasetType.VALIDATION: cls.get_val_data,
            DelightDatasetType.TEST: cls.get_test_data,
            DelightDatasetType.P_TRAIN: cls.get_production_train_data,
            DelightDatasetType.P_VAL: cls.get_production_val_data,
        }

        return enum[datatype](options)

    @classmethod
    def get_train_data(
        cls, options: DelightDatasetOptions
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Given an option, returns train dataset.

        Attributes:
        - options: A object that defines data location and transformations to use.

        Returns:
        - Tuple of two numpy arrays (X, y)
        """
        nlevels = options.n_levels
        ifold = options.fold
        domask = options.mask
        doobject = options.object
        source = options.source
        balance = options.balance

        id_train: npt.NDArray[np.str_] = np.load(
            os.path.join(
                source,
                f"id_train_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            ),
            allow_pickle=True,
        )
        y_train: npt.NDArray[np.float32] = np.load(
            os.path.join(
                source,
                f"y_train_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )
        X_train: npt.NDArray[np.float32] = np.load(
            os.path.join(
                source,
                f"X_train_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )

        if balance is False:
            return X_train, y_train

        # TODO: desacoplar oid

        idxAsiago = np.array(
            [i for i in range(id_train.shape[0]) if id_train[i][:2] == "SN"]
        )
        idxZTF = np.array(
            [i for i in range(id_train.shape[0]) if id_train[i][:3] == "ZTF"]
        )
        nimb = int(idxZTF.shape[0] / idxAsiago.shape[0])

        idxbal = np.array([], dtype=int)
        for i in range(nimb + 1):
            idxbal = np.concatenate([idxbal, idxAsiago])
            idxbal = np.concatenate(
                [
                    idxbal,
                    idxZTF[
                        i * idxAsiago.shape[0] : min(
                            idxZTF.shape[0], (i + 1) * idxAsiago.shape[0]
                        )
                    ],
                ]
            )
        np.random.shuffle(idxbal)

        id_train = id_train[idxbal]
        X_train = X_train[idxbal]
        y_train = y_train[idxbal]

        return X_train, y_train

    @classmethod
    def get_val_data(
        cls, options: DelightDatasetOptions
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Given an option, returns validation dataset.

        Attributes:
        - options: A object that defines data location and transformations to use.

        Returns:
        - Tuple of two numpy arrays (X, y)
        """
        nlevels = options.n_levels
        ifold = options.fold
        domask = options.mask
        doobject = options.object
        source = options.source
        pixscale = 0.25

        id_val: npt.NDArray[np.str_] = np.load(
            os.path.join(
                source,
                f"id_val_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            ),
            allow_pickle=True,
        )
        y_val: npt.NDArray[np.float32] = np.load(
            os.path.join(
                source,
                f"y_val_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )
        X_val: npt.NDArray[np.float32] = np.load(
            os.path.join(
                source,
                f"X_val_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )

        distance = np.sqrt(np.sum(y_val**2, axis=1))
        mask = (distance * pixscale) < 60
        X_val = X_val[mask]
        y_val = y_val[mask]
        id_val = id_val[mask]

        return X_val, y_val

    @classmethod
    def get_test_data(
        cls, options: DelightDatasetOptions
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Given an option, returns test dataset.

        Attributes:
        - options: A object that defines data location and transformations to use.

        Returns:
        - Tuple of two numpy arrays (X, y)
        """
        nlevels = options.n_levels
        domask = options.mask
        doobject = options.object
        source = options.source

        y_test = np.load(
            os.path.join(
                source, f"y_test_nlevels{nlevels}_mask{domask}_objects{doobject}.npy"
            )
        )
        X_test = np.load(
            os.path.join(
                source, f"X_test_nlevels{nlevels}_mask{domask}_objects{doobject}.npy"
            )
        )

        return X_test, y_test

    @classmethod
    def get_production_train_data(
        cls, options: DelightDatasetOptions
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Given an option, returns production train dataset, i.e, train and validation dataset concatenated.

        Attributes:
        - options: A object that defines data location and transformations to use.

        Returns:
        - Tuple of two numpy arrays (X, y)
        """
        source = options.source
        nlevels = options.n_levels
        ifold = options.fold
        domask = options.mask
        doobject = options.object

        y_train = np.load(
            os.path.join(
                source,
                f"y_train_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )
        X_train = np.load(
            os.path.join(
                source,
                f"X_train_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )
        y_val = np.load(
            os.path.join(
                source,
                f"y_val_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )
        X_val = np.load(
            os.path.join(
                source,
                f"X_val_nlevels{nlevels}_fold{ifold}_mask{domask}_objects{doobject}.npy",
            )
        )

        X_train = np.concatenate([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])

        return X_train, y_train

    @classmethod
    def get_production_val_data(
        cls, options: DelightDatasetOptions
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Given an option, returns production validation dataset, i.e, test dataset.

        Attributes:
        - options: A object that defines data location and transformations to use.

        Returns:
        - Tuple of two numpy arrays (X, y)
        """
        return cls.get_test_data(options)

    @staticmethod
    def transform(
        y: np.ndarray[Any, np.dtype[np.float32]], rot: bool, flip: bool
    ) -> torch.Tensor:
        """Transforms label vector with rotations and flips.
        For each label in `y`, transformation works as follows:
        - If `rot` is enabled, then the label is rotated in 90, 180 and 270 degrees.
        - If `flip` is enabled, then the label is flipped horizontally.

        Notice that rotations are applied also in flipped vectors. Then, if `rot` and `flip`
        are enabled, then from one label, eight vectors will be generated, where:
        - 1 vector is the original label
        - 3 vectors are rotations (90, 180, 270) of the original label
        - 1 vector is the horizontal flipped version of the original label.
        - 3 vectors are rotations (90, 180, 270) of the horizontal flipped vector.

        Attributes:
        - y: A numpy array of labels.
        - rot: Flag that create rotations of `y`.
        - flip: Flag that flips `y` vector

        Returns:
        - A data augmentation of `y`.
        """

        transforms: tuple[np.ndarray[Any, np.dtype[np.float32]], ...]

        if rot is False and flip is False:
            return torch.Tensor(y)

        if rot is False:
            yflip = cast(np.ndarray[Any, np.dtype[np.float32]], [1, -1] * y)
            transforms = (y, yflip)

        elif flip is False:
            y90 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y[:, ::-1])
            y180 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y90[:, ::-1])
            y270 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y180[:, ::-1])
            transforms = (y, y90, y180, y270)

        else:
            y90 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y[:, ::-1])
            y180 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y90[:, ::-1])
            y270 = cast(np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * y180[:, ::-1])
            yflip = cast(np.ndarray[Any, np.dtype[np.float32]], [1, -1] * y)
            yflip90 = cast(
                np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * yflip[:, ::-1]
            )
            yflip180 = cast(
                np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * yflip90[:, ::-1]
            )
            yflip270 = cast(
                np.ndarray[Any, np.dtype[np.float32]], [-1, 1] * yflip180[:, ::-1]
            )
            transforms = (y, y90, y180, y270, yflip, yflip90, yflip180, yflip270)

        return torch.Tensor(np.concatenate(transforms, axis=1))

    @staticmethod
    def detransform(
        y: npt.NDArray[np.float32], rot: bool, flip: bool
    ) -> npt.NDArray[np.float32]:
        """Transforms the prediction vector removing rotations and flips.

        Attributes:
        - y: A numpy array of predictions.

        Returns:
        - A vector of predictions without rotations and flips
        """
        nbatchs = y.shape[0]
        if rot is False and flip is False:
            return y.reshape((nbatchs, 1, 2))

        if rot is False:
            _y = y.reshape((nbatchs, 2, 2))[:, 0]
            yflip = y.reshape((nbatchs, 2, 2))[:, 1, :] * [1, -1]
            return np.dstack([_y, yflip]).reshape((nbatchs, 2, 2)).swapaxes(1, 2)

        if flip is False:
            _y = y.reshape((nbatchs, 4, 2))[:, 0]
            y90 = y.reshape((nbatchs, 4, 2))[:, 1, ::-1] * [1, -1]
            y180 = y.reshape((nbatchs, 4, 2))[:, 2, :] * [-1, -1]
            y270 = y.reshape((nbatchs, 4, 2))[:, 3, ::-1] * [-1, 1]
            return (
                np.dstack([_y, y90, y180, y270]).reshape((nbatchs, 2, 4)).swapaxes(1, 2)
            )

        return (
            np.dstack(
                [
                    y.reshape((nbatchs, 8, 2))[:, 0],
                    y.reshape((nbatchs, 8, 2))[:, 1, ::-1] * [1, -1],
                    y.reshape((nbatchs, 8, 2))[:, 2, :] * [-1, -1],
                    y.reshape((nbatchs, 8, 2))[:, 3, ::-1] * [-1, 1],
                    y.reshape((nbatchs, 8, 2))[:, 4, :] * [1, -1],
                    y.reshape((nbatchs, 8, 2))[:, 5, ::-1],
                    y.reshape((nbatchs, 8, 2))[:, 6, :] * [-1, 1],
                    y.reshape((nbatchs, 8, 2))[:, 7, ::-1] * [-1, -1],
                ]
            )
            .reshape((nbatchs, 2, 8))
            .swapaxes(1, 2)
        )

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
        """Utility function that returns the dataset in Tensorflow tensors.

        Returns:
        - A tuple of Tensorflow tensors (X, y).
        """
        X = cast(np.ndarray[Any, np.dtype[np.float32]], self.X.numpy())  # type: ignore
        y = cast(np.ndarray[Any, np.dtype[np.float32]], self.y.numpy())  # type: ignore

        return tnp.copy(X.transpose((0, 2, 3, 1))), tnp.copy(y)  # type: ignore
