from functools import reduce

import torch


class RotationAndFlipLayer(torch.nn.Module):
    def __init__(self, rot: bool = True, flip: bool = True):
        super().__init__()  # type: ignore
        self.metadata = None
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

        if self.rot is False:
            flipped: torch.Tensor = x.flip(dims=(h_dim,))
            transforms = (x, flipped)

        elif self.flip is False:
            rot90: torch.Tensor = x.rot90(k=1, dims=(w_dim, h_dim))
            rot180: torch.Tensor = x.rot90(k=2, dims=(w_dim, h_dim))
            rot270: torch.Tensor = x.rot90(k=3, dims=(w_dim, h_dim))
            transforms = (x, rot90, rot180, rot270)

        else:
            rot90: torch.Tensor = x.rot90(k=1, dims=(w_dim, h_dim))
            rot180: torch.Tensor = x.rot90(k=2, dims=(w_dim, h_dim))
            rot270: torch.Tensor = x.rot90(k=3, dims=(w_dim, h_dim))
            flipped: torch.Tensor = x.flip(dims=(h_dim,))
            flipped_rot90: torch.Tensor = flipped.rot90(k=1, dims=(w_dim, h_dim))
            flipped_rot180: torch.Tensor = flipped.rot90(k=2, dims=(w_dim, h_dim))
            flipped_rot270: torch.Tensor = flipped.rot90(k=3, dims=(w_dim, h_dim))
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
