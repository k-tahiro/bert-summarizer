import torch


def tile(x: torch.Tensor, count: int, dim: int = 0) -> torch.Tensor:
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .repeat(1, count) \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
