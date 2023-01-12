import logging
from itertools import product
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def list_mle(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None,
             reduction: Optional[str] = 'mean', eps: Optional[float] = 1e-10) -> torch.Tensor:
    """ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".

    Args:
        y_pred: (N, L) predictions from the model
        y_true: (N, L) ground truth labels
        mask: (N, L) 1 for available position, 0 for masked position
        reduction: 'none' | 'mean' | 'sum'
        eps: epsilon value, used for numerical stability
    Returns:
        torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    # shuffle for randomized tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    shuffled_y_pred = y_pred[:, random_indices]
    shuffled_y_true = y_true[:, random_indices]
    shuffled_mask = mask[:, random_indices] if mask is not None else None

    sorted_y_true, rank_true = shuffled_y_true.sort(descending=True, dim=1)
    y_pred_in_true_order = shuffled_y_pred.gather(dim=1, index=rank_true)
    if shuffled_mask is not None:
        y_pred_in_true_order = y_pred_in_true_order - 10000.0 * (1.0 - shuffled_mask)

    max_y_pred, _ = y_pred_in_true_order.max(dim=1, keepdim=True)
    y_pred_in_true_order_minus_max = y_pred_in_true_order - max_y_pred
    cum_sum = y_pred_in_true_order_minus_max.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    observation_loss = torch.log(cum_sum + eps) - y_pred_in_true_order_minus_max
    if shuffled_mask is not None:
        observation_loss[shuffled_mask == 0] = 0.0
    loss = observation_loss[:, :-1].sum(dim=1)
    # loss = observation_loss.sum(dim=1)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


def pairwise_hinge(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   margin: Optional[float] = 0., reduction: Optional[str] = 'mean') -> torch.Tensor:
    """RankNet loss introduced in "Learning to Rank using Gradient Descent".

    Args:
        y_pred: (N, L) predicted scores
        y_true: (N, L) ground truth labels
        mask: (N, L) 1 for available position, 0 for masked position
        margin:
        reduction: 'none' | 'mean' | 'sum'
    Returns:
        torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    if mask is not None:
        y_pred = y_pred.clone()
        y_true = y_true.clone()
        y_pred[mask == 0] = float('-inf')
        y_true[mask == 0] = float('-inf')

    # generate every pair of indices from the range of candidates number in the batch
    candidate_pairs = list(product(range(y_true.shape[1]), repeat=2))  # (L^2, 2)

    # (N, L^2, 2)
    pairs_true = y_true[:, candidate_pairs]
    pairs_pred = y_pred[:, candidate_pairs]

    # calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]  # (N, L^2)

    # filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symmetric pairs, so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))  # (N, L^2)

    # (num_pairs,)
    s1 = pairs_pred[:, :, 0][the_mask]
    s2 = pairs_pred[:, :, 1][the_mask]
    target = the_mask.float()[the_mask]

    # (N, L^2)
    pair_losses = torch.zeros_like(pairs_pred[:, :, 0])
    pair_losses[the_mask] = F.margin_ranking_loss(s1, s2, target, margin=margin, reduction='none')
    # pair_losses[the_mask] = (s2 - s1 + margin).relu()

    # (N,)
    loss = pair_losses.sum(dim=1) / the_mask.sum(dim=1)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()
