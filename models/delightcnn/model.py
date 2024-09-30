import math
from collections import OrderedDict

import torch

from .layers import RotationAndFlipLayer
from .schemas import DelightCnnParameters


class DelightCnn(torch.nn.Module):
    def __init__(self, options: DelightCnnParameters, channels: int = 1):
        super().__init__()  # type: ignore
        bottleneck: OrderedDict[str, torch.nn.Module] = OrderedDict(
            [
                ("conv1", torch.nn.Conv2d(channels, options.nconv1, 3)),
                ("relu1", torch.nn.ReLU()),
                ("mp1", torch.nn.MaxPool2d(2)),
                ("conv2", torch.nn.Conv2d(options.nconv1, options.nconv2, 3)),
                ("relu2", torch.nn.ReLU()),
                ("mp2", torch.nn.MaxPool2d(2)),
                ("conv3", torch.nn.Conv2d(options.nconv2, options.nconv3, 3)),
                ("relu3", torch.nn.ReLU()),
                ("flatten", torch.nn.Flatten()),
            ]
        )
        regression: OrderedDict[str, torch.nn.Module] = OrderedDict(
            [
                (
                    "input_layer_ffcc",
                    torch.nn.Linear(
                        in_features=self._compute_dense_features(
                            levels=options.levels, bottleneck=bottleneck
                        ),
                        out_features=options.ndense,
                    ),
                ),
                ("tanh", torch.nn.Tanh()),
                ("dropout", torch.nn.Dropout(p=options.dropout)),
                ("out", torch.nn.Linear(in_features=options.ndense, out_features=2)),
            ]
        )

        self.rot_and_flip = RotationAndFlipLayer(rot=options.rot, flip=options.flip)
        self.bottleneck = torch.nn.Sequential(bottleneck)
        self.regression = torch.nn.Sequential(regression)

    @staticmethod
    def _compute_dense_features(
        bottleneck: OrderedDict[str, torch.nn.Module],
        levels: int,
    ) -> int:
        w = 30
        h = 30
        conv_out = 0
        for layer in bottleneck.values():
            if isinstance(layer, torch.nn.Conv2d):
                k = layer.kernel_size[0]
                w = w - k + 1
                h = h - k + 1
                conv_out = layer.out_channels
            if isinstance(layer, torch.nn.MaxPool2d):
                kw, kh = (
                    layer.kernel_size
                    if isinstance(layer.kernel_size, tuple)
                    else (layer.kernel_size, layer.kernel_size)
                )
                w = math.floor((w - kw) / 2 + 1)
                h = math.floor((h - kh) / 2 + 1)

        return w * h * conv_out * levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        x = self.rot_and_flip(x)
        x = self.bottleneck(x)

        x = x.reshape(batch, self.rot_and_flip.n_transforms, -1)

        x = self.regression(x)

        return x.reshape(batch, self.rot_and_flip.n_transforms * 2)
