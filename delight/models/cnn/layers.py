from functools import reduce

import torch


class RotationAndFlipLayer(torch.nn.Module):
    def __init__(self, rot: bool = True, flip: bool = True):
        super().__init__()  # type: ignore
        self.rot = rot
        self.flip = flip
        self.n_transforms = (int(flip) + 1) * (3 * int(rot) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 40 x 5 x 1 x 30 x 30
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
                x,                # 2 x 5 x 1 x 30 x 30
                rot90,            # 2 x 5 x 1 x 30 x 30
                rot180,           # 2 x 5 x 1 x 30 x 30
                rot270,           # 2 x 5 x 1 x 30 x 30
                flipped,          # 2 x 5 x 1 x 30 x 30
                flipped_rot90,    # 2 x 5 x 1 x 30 x 30
                flipped_rot180,   # 2 x 5 x 1 x 30 x 30
                flipped_rot270,   # 2 x 5 x 1 x 30 x 30
            )

        # 2 x 40 x 1 x 30 x 30
        x = torch.cat(transforms, dim=1)
        
        # 80 x 1 x 30 x 30
        return x.reshape(
            stacked * self.n_transforms, x.shape[-3], x.shape[-2], x.shape[-1]
        )
