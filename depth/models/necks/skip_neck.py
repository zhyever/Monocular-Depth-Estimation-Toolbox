# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init

from depth.ops import resize
from depth.models.builder import NECKS


@NECKS.register_module()
class SkipNeck(nn.Module):
    """SkipNeck.

    Reshape the skip features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=[0.5, 1, 2, 4]):
        super(SkipNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        for i in range(self.num_outs):
            if self.scales[i] != 1:
                x_resize = resize(
                    inputs[i], scale_factor=self.scales[i], mode='bilinear')
            else:
                x_resize = inputs[i]
            outs.append(x_resize)
        return tuple(outs)
