import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F
from depth.models.builder import build_loss
from depth.ops import resize
from depth.models.decode_heads import DenseDepthHead

class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))


class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps
        
class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x

class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)

@HEADS.register_module()
class AdabinsHead(DenseDepthHead):
    """AdaBins: Depth Estimation using Adaptive Bins.
    This head is implemented of `Adabins: <https://arxiv.org/abs/2011.14141>`_.
    Args:
        n_bins (int): The number of bins used in cls.-reg. Default: 256.
        patch_size (int): The number of patches in mini-ViT. Default: 16.
        loss_chamfer (dict): charmfer loss for supervision on bins.
            Default: dict(type='BinsChamferLoss').
    """

    def __init__(self,
                 n_bins=256,
                 patch_size=16,
                 loss_chamfer=dict(type='BinsChamferLoss', loss_weight=0.1),
                 **kwargs):
        super(AdabinsHead, self).__init__(**kwargs)

        self.loss_chamfer = build_loss(loss_chamfer)

        self.conv_list = nn.ModuleList()
        up_channel_temp = 0
        for index, (in_channel, up_channel) in enumerate(
                zip(self.in_channels, self.up_sample_channels)):
            if index == 0:
                self.conv_list.append(
                    ConvModule(
                        in_channels=in_channel,
                        out_channels=up_channel,
                        kernel_size=1,
                        stride=1,
                        act_cfg=None
                    ))
            else:
                self.conv_list.append(
                    UpSample(skip_input=in_channel + up_channel_temp,
                             output_features=up_channel,
                             norm_cfg=self.norm_cfg,
                             act_cfg=self.act_cfg))

            # save earlier fusion target
            up_channel_temp = up_channel


        # final self.channels = 128
        self.n_bins = n_bins

        self.decode_final_conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)

        self.adaptive_bins_layer = mViT(self.channels,
                                        n_query_channels=self.channels,
                                        patch_size=patch_size,
                                        dim_out=n_bins,
                                        embedding_dim=self.channels,
                                        norm='linear')

        self.conv_out = nn.Sequential(nn.Conv2d(self.channels, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1)) # defualt channels=128


    def forward(self, inputs, img_metas):
        """Forward function."""
        # inputs order first -> end
        temp_feat_list = []
        for index, feat in enumerate(inputs[::-1]):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
                temp_feat_list.append(temp_feat)
            else:
                skip_feat = feat
                up_feat = temp_feat_list[index-1]
                temp_feat = self.conv_list[index](up_feat, skip_feat)
                temp_feat_list.append(temp_feat)

        decode_out_feat = self.decode_final_conv(temp_feat_list[-1])

        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(decode_out_feat)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_depth - self.min_depth) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)

        output = torch.sum(out * centers, dim=1, keepdim=True)

        return output, bin_edges


    def forward_train(self, img, inputs, img_metas, depth_gt, train_cfg):
        depth_pred, bin_edges = self.forward(inputs, img_metas)
        depth_pred = resize(
            input=depth_pred,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)

        losses = dict()
        losses["loss_depth"] = self.loss_decode(depth_pred, depth_gt)
        losses["loss_chamfer"] = self.loss_chamfer(bin_edges, depth_gt)

        log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
        losses.update(**log_imgs)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):

        depth_pred, bin_edges = self.forward(inputs, img_metas)
        return depth_pred

