from typing import List

import torch


def mask_where0(x, m):
    """

    Args:
        x (torch.Tensor): (*)
        m (torch.Tensor): same size as logits
            1 for positions that are NOT MASKED, 0 for MASKED positions.

    Returns:
        torch.Tensor: same size as logits
    """
    if x.dtype == torch.float16:
        return x * m - 65500 * (1 - m)
    else:
        return x * m - 1e30 * (1 - m)


def pad_tensors(tensors: List[torch.Tensor], pad_val: int, pad_to_multiple_of: int = 1,
                left_pad: bool = False, move_eos_to_beginning: bool = False, eos_val: bool = None):
    """Convert a list of 1d tensors into a padded 2d tensor."""

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_val
            dst[0] = eos_val
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if len(tensors[0].size()) > 1:
        tensors = [x.view(-1) for x in tensors]
    batch_size = len(tensors)
    max_len = max(x.size(0) for x in tensors)
    if max_len % pad_to_multiple_of != 0:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of
    padded_tensor = tensors[0].new_full((batch_size, max_len), pad_val, requires_grad=tensors[0].requires_grad)
    for i, x in enumerate(tensors):
        copy_tensor(x, padded_tensor[i, max_len - len(x):] if left_pad else padded_tensor[i, :len(x)])
    return padded_tensor
