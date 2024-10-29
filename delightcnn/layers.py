from functools import reduce

import torch


class RotationAndFlipLayer(torch.nn.Module):
    """Custom layer that transform an input into a stack of rotations and flips.

    It assumes an input of, at least, (B, L, C, W, H) where:
    - B: Batch
    - L: Levels
    - C: Channels
    - W: Width
    - H: Height

    For each element in the input, the transformation works as follows:
    - If `rot` is enabled, then the element is rotated in 90, 180 and 270 degrees.
    - If `flip` is enabled, then the element is flipped horizontally.

    Notice that rotations are applied also in flipped vectors. Then, if `rot` and `flip`
    are enabled, then from one element, eight vectors will be generated, where:
    - 1 vector is the original element
    - 3 vectors are rotations (90, 180, 270) of the original element
    - 1 vector is the horizontal flipped version of the original element.
    - 3 vectors are rotations (90, 180, 270) of the horizontal flipped vector.

    It will returns a new tensor of (B * L * T, C, W, H)
    Where T = (`flip==True` + 1) * (3 * `rot==True` + 1) 
    """
    def __init__(self, rot: bool = True, flip: bool = True):
        super().__init__()  # type: ignore
        self.rot = rot
        self.flip = flip
        self.n_transforms = (int(flip) + 1) * (3 * int(rot) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stacked = reduce(lambda x, y: x * y, x.shape[:-3], 1)

        if self.rot is False and self.flip is False:
            x = x.reshape(stacked, x.shape[-3], x.shape[-2], x.shape[-1])
            return x

        w_dim = len(x.shape) - 2
        h_dim = len(x.shape) - 1
        transforms: tuple[torch.Tensor, ...]

        if self.rot is False:
            flipped = x.flip(dims=(h_dim,))
            transforms = (x, flipped)

        elif self.flip is False:
            rot90 = x.rot90(k=1, dims=(w_dim, h_dim))
            rot180 = x.rot90(k=2, dims=(w_dim, h_dim))
            rot270 = x.rot90(k=3, dims=(w_dim, h_dim))
            transforms = (x, rot90, rot180, rot270)

        else:
            rot90 = x.rot90(k=1, dims=(w_dim, h_dim))
            rot180 = x.rot90(k=2, dims=(w_dim, h_dim))
            rot270 = x.rot90(k=3, dims=(w_dim, h_dim))
            flipped = x.flip(dims=(h_dim,))
            flipped_rot90 = flipped.rot90(k=1, dims=(w_dim, h_dim))
            flipped_rot180 = flipped.rot90(k=2, dims=(w_dim, h_dim))
            flipped_rot270 = flipped.rot90(k=3, dims=(w_dim, h_dim))
            transforms = (
                x,
                rot90,
                rot180,
                rot270,
                flipped,
                flipped_rot90,
                flipped_rot180,
                flipped_rot270,
            )

        x = torch.cat(transforms, dim=1)

        return x.reshape(
            stacked * self.n_transforms, x.shape[-3], x.shape[-2], x.shape[-1]
        )
