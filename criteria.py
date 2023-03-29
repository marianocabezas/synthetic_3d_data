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
DSC based losses
"""


def dsc_binary_loss(pred, target):
    pred = torch.flatten(pred >= 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
            2 * torch.sum(pred & target, dim=1)
    ).type(torch.float32).to(pred.device)
    sum_pred = torch.sum(pred, dim=1).type(torch.float32).to(pred.device)
    sum_target = torch.sum(target, dim=1).type(torch.float32).to(pred.device)

    dsc_k = intersection / (sum_pred + sum_target)
    dsc_k = dsc_k[torch.logical_not(torch.isnan(dsc_k))]
    if len(dsc_k) > 0:
        dsc = 1 - torch.mean(dsc_k)
    else:
        dsc = torch.tensor(0)

    return torch.clamp(dsc, 0., 1.)


def tp_binary_loss(pred, target):
    pred = torch.flatten(pred >= 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
        torch.sum(pred & target, dim=1)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(target, dim=1).type(torch.float32).to(pred.device)

    tp_k = intersection / sum_target
    tp_k = tp_k[torch.logical_not(torch.isnan(tp_k))]
    tp = 1 - torch.mean(tp_k) if len(tp_k) > 0 else torch.tensor(0)

    return torch.clamp(tp, 0., 1.)


def tn_binary_loss(pred, target):
    pred = torch.flatten(pred < 0.5, start_dim=1).to(pred.device)
    target = torch.flatten(target, start_dim=1).type_as(pred).to(pred.device)

    intersection = (
            torch.sum(pred & torch.logical_not(target), dim=1)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(
        torch.logical_not(target), dim=1
    ).type(torch.float32).to(pred.device)

    tn_k = intersection / sum_target
    tn_k = tn_k[torch.logical_not(torch.isnan(tn_k))]
    tn = 1 - torch.mean(tn_k)
    if torch.isnan(tn):
        tn = torch.tensor(0.)

    return torch.clamp(tn, 0., 1.)


def accuracy(predicted, target):
    """

    :param predicted:
    :param target:
    :return:
    """
    correct_positive = predicted * target
    correct_negative = (1. - predicted) * (1. - target)

    return torch.mean(correct_positive + correct_negative)