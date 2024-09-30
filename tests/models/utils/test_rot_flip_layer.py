import torch
from hypothesis import given
from hypothesis.strategies import booleans, integers

from models.delightcnn.layers import RotationAndFlipLayer


@given(
    integers(min_value=1, max_value=64),
    integers(min_value=1, max_value=10),
    integers(min_value=1, max_value=10),
    integers(min_value=1, max_value=30),
    booleans(),
    booleans(),
)
def test_consistency_rotation_flips(
    n_batch: int, n_levels: int, n_channels: int, im_size: int, rot: bool, flip: bool
):
    layer = RotationAndFlipLayer(rot=rot, flip=flip)
    x = torch.zeros((n_batch, n_levels, n_channels, im_size, im_size))
    out = layer.forward(x)

    expected = n_batch * n_levels * layer.n_transforms
    size = torch.Size([expected, n_channels, im_size, im_size])
    assert out.shape == size


def test_disable_rot_or_flip_n_transforms():
    assert RotationAndFlipLayer(rot=False, flip=False).n_transforms == 1
    assert RotationAndFlipLayer(rot=False, flip=True).n_transforms == 2
    assert RotationAndFlipLayer(rot=True, flip=False).n_transforms == 4
    assert RotationAndFlipLayer(rot=True, flip=True).n_transforms == 8
