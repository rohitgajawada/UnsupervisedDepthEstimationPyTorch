from __future__ import division
from torch import nn
from torch.autograd import Variable
# TODO: from inverse_warp import inverse_warp


def explainability_loss(mask):
    """
    Regularizer for the explainability mask. Forces non-zero mask.
    @param mask: list/tuple of explainability masks, at different scales.
    """
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        # Requires mask to be an autograd Variable
        ones = Variable(torch.ones(mask_scaled.size()))
        loss += nn.functional.binary_cross_entropy(mask, ones)
    return loss


def gradient(pred):
    """
    Returns the gradient as a tuple
    @param pred: 3D tensor
    """
    D_dy = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def smooth_loss(pred):
    """
    Multi scale smoothing loss for the image
    @param pred: list/tuple of predictions at different scales
    """
    if type(pred) not in [list, tuple]:
        pred = [pred]

    loss = 0
    weight = 1

    for pred_scaled in pred:
        dx, dy = gradient(pred_scaled)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean()
                + dydx.abs().mean() + dy2.abs().mean()) * weight
        weight /= 2.83 # 2sqrt(2)
    return loss
