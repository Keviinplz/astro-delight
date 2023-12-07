import torch
from hypothesis import given
from hypothesis.strategies import booleans, floats, integers

from delight.models.cnn import DelightCnn, DelightCnnParameters


@given(
    integers(min_value=1, max_value=10),
    integers(min_value=16, max_value=52),
    integers(min_value=32, max_value=57),
    integers(min_value=32, max_value=57),
    integers(min_value=128, max_value=685),
    integers(min_value=1, max_value=10),
    floats(min_value=0.0, max_value=1.0),
    booleans(),
    booleans(),
)
def test_delight_cnn_initialization(
    batch: int,
    nconv1: int,
    nconv2: int,
    nconv3: int,
    ndense: int,
    levels: int,
    dropout: float,
    rot: bool,
    flip: bool,
):
    x = torch.rand(batch, levels, 1, 30, 30)
    model = DelightCnn(
        DelightCnnParameters(
            nconv1=nconv1,
            nconv2=nconv2,
            nconv3=nconv3,
            ndense=ndense,
            levels=levels,
            dropout=dropout,
            rot=rot,
            flip=flip,
        )
    )

    try:
        out = model.forward(x)
    except RuntimeError:
        assert False, f"Runtime error on model with batch {batch} and levels {levels}"

    expected = torch.Size([batch, model.rot_and_flip.n_transforms * 2])
    assert (
        out.shape == expected
    ), f"Failed with batch {batch} and levels {levels} => {out.shape} != {expected}"
