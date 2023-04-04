import torch
from torch.nn import functional as F


"""
Entropy-based
"""


def focal_loss(pred, target, gamma=2.0):
    """
    Function to compute the focal loss based on:
    Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. "Focal
    Loss for Dense Object Detection".
    https://arxiv.org/abs/1708.02002
    https://ieeexplore.ieee.org/document/8237586
    :param pred: Predicted values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param target: Ground truth values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param gamma: Focusing parameter (default 2.0).
    :return: Focal loss value.
    """

    m_bg = target == 0
    m_fg = target > 0

    pt_fg = pred[m_fg]
    pt_bg = (1 - pred[m_bg])

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    bce_fg = bce[m_fg]
    bce_bg = bce[m_bg]

    focal_fg = (1 - pt_fg).pow(gamma) * bce_fg
    focal_bg = (1 - pt_bg).pow(gamma) * bce_bg

    focal = torch.cat([focal_fg, focal_bg])

    return focal.mean()


"""
Similarity losses
"""


def similarity_loss(var_pair):
    mse_losses = [F.mse_loss(xi, yi) for xi, yi in var_pair]

    return sum(mse_losses)


"""
Overlap losses
"""


def agreement_ratio(numerator, denominator, device):
    num_sum = torch.sum(numerator).to(device, torch.float32)
    den_sum = torch.sum(denominator).to(device, torch.float32)
    ratio = num_sum / den_sum
    ratio = ratio[torch.logical_not(torch.isnan(ratio))]
    neg_ratio = 1 - torch.mean(ratio) if len(ratio) > 0 else torch.tensor(0)
    return torch.clamp(neg_ratio, 0., 1.)


def tp_binary_loss(pred, target):
    pred = pred >= 0
    target = target > 0

    return agreement_ratio(pred & target, target, pred.device)


def tn_binary_loss(pred, target):
    pred = pred < 0
    target = target < 1

    return agreement_ratio(pred & target, target, pred.device)


def accuracy_loss(predicted, target):
    """

    :param predicted:
    :param target:
    :return:
    """
    predicted = predicted >= 0
    target = target > 0
    correct_positive = predicted & target
    correct_negative = torch.logical_not(predicted) & torch.logical_not(target)
    acc = 1 - torch.mean(
        torch.cat([correct_positive, correct_negative]).type(torch.float32)
    )

    return torch.clamp(acc, 0., 1.)
