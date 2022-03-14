import torch

from torch import nn

from ..builder import BACKBONES

@BACKBONES.register_module()
class EfficientNet(nn.Module):
    """EfficientNet backbone.
    Following Adabins, this is a hack version of EfficientNet, where potential bugs exist.

    Args:
        basemodel_name (str): Name of pre-trained EfficientNet. Default: tf_efficientnet_b5_ap.
        out_index List(int): Output from which stages. Default: [4, 5, 6, 8, 11].

    """
    def __init__(self, 
                 basemodel_name='tf_efficientnet_b5_ap',
                 out_index=[4, 5, 6, 8, 11]):
        super(EfficientNet, self).__init__()
        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel
        self.out_index = out_index

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))

        out = []
        for index in self.out_index:
            out.append(features[index])

        return tuple(out)