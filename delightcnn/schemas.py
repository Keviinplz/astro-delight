from dataclasses import dataclass


@dataclass
class DelightCnnParameters:
    """Defines DelightCnn parameters.

    Attributes:
    - nconv1: Number of out channels for first convolutional layer.
    - nconv2: Number of out channels for second convolutional layer.
    - nconv3: Number of out channels for third convolutional layer.
    - ndense: Number of out features for first fully-connected layer.
    - dropout: Probability of dropout. 0 means no dropout.
    - channels: Number of channels expected from the dataset.
    - levels: Quantity of levels expected from the dataset.
    - rot: Applies a rotation transformation on the input.
    - flip: Applies a flip transformation on the input.
    """

    nconv1: int
    nconv2: int
    nconv3: int
    ndense: int
    dropout: float
    channels: int
    levels: int
    rot: bool
    flip: bool
