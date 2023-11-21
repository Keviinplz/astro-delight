import math
from collections import OrderedDict
from typing import TypedDict

import torch
from ray.tune.search.sample import Float

from astro_delight.models.cnn.layers import RotationAndFlipLayer


class DelightCnnParameters(TypedDict):
    nconv1: int | float
    nconv2: int | float
    nconv3: int | float
    ndense: int | float
    levels: int
    dropout: float
    rot: bool
    flip: bool


class DelightCnnParametersRay(TypedDict):
    nconv1: Float
    nconv2: Float
    nconv3: Float
    ndense: Float
    levels: int
    dropout: Float
    rot: bool
    flip: bool


class DelightCnn(torch.nn.Module):
    def __init__(self, options: DelightCnnParameters):
        super().__init__()  # type: ignore

        nconv1 = (
            int(2 ** options["nconv1"])
            if isinstance(options["nconv1"], float)
            else options["nconv1"]
        )
        nconv2 = (
            int(2 ** options["nconv2"])
            if isinstance(options["nconv2"], float)
            else options["nconv2"]
        )
        nconv3 = (
            int(2 ** options["nconv3"])
            if isinstance(options["nconv3"], float)
            else options["nconv3"]
        )
        ndense = (
            int(2 ** options["ndense"])
            if isinstance(options["ndense"], float)
            else options["ndense"]
        )

        bottleneck: OrderedDict[str, torch.nn.Module] = OrderedDict(
            [
                ("conv1", torch.nn.Conv2d(1, nconv1, 3)),
                ("relu1", torch.nn.ReLU()),
                ("mp1", torch.nn.MaxPool2d(2)),
                ("conv2", torch.nn.Conv2d(nconv1, nconv2, 3)),
                ("relu2", torch.nn.ReLU()),
                ("mp2", torch.nn.MaxPool2d(2)),
                ("conv3", torch.nn.Conv2d(nconv2, nconv3, 3)),
                ("relu3", torch.nn.ReLU()),
                ("flatten", torch.nn.Flatten()),
            ]
        )
        linear_in = self._compute_dense_features(
            levels=options["levels"], bottleneck=bottleneck
        )

        fc: OrderedDict[str, torch.nn.Module] = OrderedDict(
            [
                (
                    "fc1",
                    torch.nn.Linear(in_features=linear_in, out_features=ndense),
                ),
                ("tanh", torch.nn.Tanh()),
                ("dropout", torch.nn.Dropout(p=options["dropout"])),
                ("fc2", torch.nn.Linear(in_features=ndense, out_features=2)),
            ]
        )

        self.rot_and_flip = RotationAndFlipLayer(
            rot=options["rot"], flip=options["flip"]
        )
        self.bottleneck = torch.nn.Sequential(bottleneck)
        self.fc = torch.nn.Sequential(fc)

    def _compute_dense_features(
        self,
        *,
        bottleneck: OrderedDict[str, torch.nn.Module],
        levels: int,
    ) -> int:
        w = 30
        h = 30
        conv_out = 0
        for layer in bottleneck.values():
            k: int
            if isinstance(layer, torch.nn.Conv2d):
                k = layer.kernel_size[0]
                w = w - k + 1
                h = h - k + 1
                conv_out = layer.out_channels
            if isinstance(layer, torch.nn.MaxPool2d):
                k = layer.kernel_size  # type: ignore
                w = math.floor((w - k) / 2 + 1)
                h = math.floor((h - k) / 2 + 1)

        return w * h * conv_out * levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]  # TODO: Remove batch dependency

        # Apply flips and rotations over level (L) dimension
        x = self.rot_and_flip(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Undo transformations
        x = x.reshape(batch, self.rot_and_flip.n_transforms, -1)

        # Linear
        x = self.fc(x)
        return x.reshape(batch, self.rot_and_flip.n_transforms * 2)
