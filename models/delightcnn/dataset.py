from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore
import torch
from torch.utils.data import Dataset


@dataclass
class DelightDatasetOptions:
    """
    Defines where to find data and how to transform it.

    Attributes:
    - rot: Flag that indicates if a rotation transformation has to be used on the data to generate a data augmentation.
    - flip: Flag that indicates if a flip transformation has to be used on the data to generate a data augmentation.
    """
    channels: int
    levels: int
    rot: bool
    flip: bool


class Processor(Protocol):
    @property
    def X(self) -> np.ndarray[Any, np.dtype[np.float32]]: ...

    @property
    def y(self) -> np.ndarray[Any, np.dtype[np.float32]]: ...


class DelightDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, processor: Processor, options: DelightDatasetOptions | None = None
    ):
        """
        Initialize an instance of a `torch.utils.data.Dataset` object.

        Attributes:
        - procesor: An Processor object that defines where to get data, and has methods X and y to retrieve processed data.
        - options: An object that defines data location and transformations to use.
        """

        self.X = torch.Tensor(processor.X)
        self.y = (
            self.transform(
                processor.y,
                options.rot,
                options.flip,
            )
            if options is not None
            else torch.from_numpy(processor.y)  # type: ignore
        )

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
        """Detransforms the prediction vector removing rotations and flips.

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
