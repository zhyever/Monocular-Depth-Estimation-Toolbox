# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

from depth.models.builder import LOSSES


@LOSSES.register_module()
class BinsChamferLoss(nn.Module):
    """BinsChamferLoss used in Adabins. Waiting for re-writing

    Args:
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 loss_weight=0.1):
        super(BinsChamferLoss, self).__init__()
        self.loss_weight = loss_weight

    def bins_chamfer_loss(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

    def forward(self,
                input,
                target,
                **kwargs):
        """Forward function."""
        
        chamfer_loss = self.bins_chamfer_loss(input, target)
        chamfer_loss = self.loss_weight * chamfer_loss
        return chamfer_loss
