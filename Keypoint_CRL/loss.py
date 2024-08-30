import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_joint_si_loss(output, target):
    batch, num_points, _, _ = output.shape
    if output.shape != target.shape:
        # target = nn.functional.interpolate(target, size=output.shape[2:], mode='bilinear', align_corners=True)
        output = nn.functional.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=True)
    loss = 0.1 * F.l1_loss(output[target < 1e-6], target[target < 1e-6], reduction='mean')
    loss += 0.8 * F.l1_loss(output[target >= 1e-6], target[target >= 1e-6], reduction='mean')
    g = target - output
    loss += 0.1 * torch.var(g)
    return loss


def JointsSILoss(output, target):
    if isinstance(output, list) or isinstance(output, tuple):
        print(type(output))
        loss = 0
        for i in range(len(output)):
            loss += cal_joint_si_loss(output[i], target)
        return loss
    else:
        return cal_joint_si_loss(output, target)
