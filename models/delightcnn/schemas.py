from dataclasses import dataclass


@dataclass
class DelightCnnParameters:
    """Defines DelightCnn parameters.

    Attributes:
    - nconv1: Number of out channels for first convolutional layer.
    - nconv2: Number of out channels for second convolutional layer.
    - nconv3: Number of out channels for third convolutional layer.
    - ndense: Number of out features for first fully-connected layer.
    - levels: Quantity of levels expected from the dataset.
    - levels: Probability of dropout. 0 means no dropout.
    - rot: Applies a rotation transformation on the input.
    - flip: Applies a flip transformation on the input.
    """

    nconv1: int
    nconv2: int
    nconv3: int
    ndense: int
    levels: int
    dropout: float
    rot: bool
    flip: bool
