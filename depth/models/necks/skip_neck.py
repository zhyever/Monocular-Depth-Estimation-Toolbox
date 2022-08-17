# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init

from depth.ops import resize
from depth.models.builder import NECKS


@NECKS.register_module()
class SkipNeck(nn.Module):
    """SkipNeck.

        Reshape the skip features. Simple hack of reassemble layer in `DPT <https://arxiv.org/abs/2103.13413>`_.

    Args:
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 reassemble=False,
                 scales=[0.5, 1, 2, 4]):
        super(SkipNeck, self).__init__()
        self.scales = scales
        self.num_outs = len(scales)
        self.reassemble = reassemble

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        outs = []
        for i in range(self.num_outs):

            if self.reassemble:
                x, cls_token = inputs[i]
            else:
                x = inputs[i]

            if self.scales[i] != 1:
                x_resize = resize(
                    x, scale_factor=self.scales[i], mode='bilinear', align_corners=True)
            else:
                x_resize = x
                
            outs.append(x_resize)

        return tuple(outs)
