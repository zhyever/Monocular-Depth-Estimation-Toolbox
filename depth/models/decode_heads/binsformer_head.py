# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init

from depth.ops import resize
from ..builder import HEADS
from .decode_head import DepthBaseDecodeHead

from mmcv.runner import BaseModule, auto_fp16
from depth.models.utils.builder import build_transformer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding, build_attention
import torch.nn.functional as F
import numpy as np
from depth.models.builder import build_loss

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

@HEADS.register_module()
class BinsFormerDecodeHead(DepthBaseDecodeHead):
    """BinsFormer head
    This head is implemented of `BinsFormer: <https://arxiv.org/abs/2204.00987>`_.
    Motivated by segmentation methods, we design a double-stream decoders to achieve depth estimation.
    Args:
        binsformer (bool): Switch from the baseline method to Binsformer module. Default: False.
        align_corners (bool): Whether to apply align_corners mode to achieve upsample. Default: True.
        norm_cfg (dict|): Config of norm layers.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config of activation layers.
            Default: dict(type='LeakyReLU', inplace=True).
        dms_decoder (bool): Whether to apply a transfomer encoder before cross-attention with queries. Default: True.
        transformer_encoder (dict|None): General transfomer encoder config before cross-attention with queries. 
        positional_encoding (dict|None): Position encoding (p.e.) config.
        conv_dim (int): Temp feature dimension. Default: 256.
        index (List): Default indexes of input features from encoder/neck module. Default: [0,1,2,3,4]
        trans_index (List): Selected indexes of pixel-wise features to apply self-/cross- attention with transformer head.
        transformer_decoder(dict|None): Config of transformer decoder.
        with_loss_chamfer (bool): Whether to apply chamfer loss on bins distribution. Default: False
        loss_chamfer (dict|None): Config of the chamfer loss.
        classify (bool): Whether to apply scene understanding aux task. Default: True.
        class_num (int): class number for scene understanding aux task. Default: 25
        loss_class (dict): Config of scene classification loss. Default: dict(type='CrossEntropyLoss', loss_weight=1e-1).
        train_cfg (dict): Config of aux loss following most detr-like methods.
            Default: dict(aux_loss=True,),
    """
    def __init__(self,
                 binsformer=True,
                 align_corners=True,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 dms_decoder=True,
                 transformer_encoder=None,
                 positional_encoding=None,
                 conv_dim=256,
                 index=[0,1,2,3,4],
                 trans_index=[1,2,3],
                 transformer_decoder=None,
                 with_loss_chamfer=False,
                 loss_chamfer=None,
                 classify=True,
                 class_num=25,
                 loss_class=dict(type='CrossEntropyLoss', loss_weight=1e-1),
                 train_cfg=dict(aux_loss=True,),
                 **kwargs):
        super(BinsFormerDecodeHead, self).__init__(**kwargs)

        self.conv_dim = conv_dim
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.transformer_encoder = build_transformer(transformer_encoder)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer_num_feature_levels = len(trans_index)
        self.index = index
        self.trans_index = trans_index
        self.level_embed = nn.Embedding(self.transformer_num_feature_levels, conv_dim)
        self.with_loss_chamfer = with_loss_chamfer
        if with_loss_chamfer:
            self.loss_chamfer = build_loss(loss_chamfer)
        
        self.train_cfg = train_cfg
        self.dms_decoder = dms_decoder

        # DMSTransformer used to apply self-att before cross-att following detr-like methods
        self.skip_proj = nn.ModuleList()
        trans_channels = [self.in_channels[i] for i in self.trans_index]
        for trans_channel in trans_channels:
            self.skip_proj.append(
                ConvModule(trans_channel,
                           self.conv_dim,
                           kernel_size=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))

        # pixel-wise decoder (FPN)
        self.num_fpn_levels = len(self.trans_index)
        lateral_convs = nn.ModuleList()
        output_convs = nn.ModuleList()

        for idx, in_channel in enumerate(self.in_channels[:self.num_fpn_levels]):
            lateral_conv = ConvModule(
                in_channel, 
                conv_dim, 
                kernel_size=1, 
                norm_cfg=norm_cfg)
            output_conv = ConvModule(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        # build queries for transformer head
        self.classify = classify
        if self.classify is True:
            self.loss_class = build_loss(loss_class)
            # transformer_decoder['decoder']['classify'] = True
            # transformer_decoder['decoder']['class_num'] = class_num
            transformer_decoder['classify'] = True
            transformer_decoder['class_num'] = class_num
            # Add an additional query for classifing.
            self.query_feat = nn.Embedding(self.n_bins + 1, conv_dim)
            self.query_embed = nn.Embedding(self.n_bins + 1, conv_dim)
        else:
            # transformer_decoder['decoder']['classify'] = False
            transformer_decoder['classify'] = False
            # learnable query features
            self.query_feat = nn.Embedding(self.n_bins, conv_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(self.n_bins, conv_dim)
            
        self.transformer_decoder = build_transformer(transformer_decoder)

        # regression baseline
        self.binsformer = binsformer
        if binsformer is False:
            self.pred_depth = ConvModule(
                self.n_bins,
                1,
                kernel_size=3,
                stride=1,
                padding=1)

        # used in visualization
        self.hook_identify_center = torch.nn.Identity()
        self.hook_identify_prob = torch.nn.Identity()
        self.hook_identify_depth = torch.nn.Identity()

    def init_weights(self):
        """Initialize weights of the Binsformer head."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.transformer_encoder.init_weights()
        self.transformer_decoder.init_weights()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        # NOTE: first apply self-att before cross-att
        out = []
        if self.dms_decoder:
            
            # projection of the input features
            trans_feats = [inputs[i] for i in self.trans_index]

            mlvl_feats = [
                skip_proj(trans_feats[i])
                for i, skip_proj in enumerate(self.skip_proj)
            ]

            batch_size = mlvl_feats[0].size(0)
            input_img_h, input_img_w = mlvl_feats[0].size(2), mlvl_feats[0].size(3)
            img_masks = mlvl_feats[0].new_zeros(
                (batch_size, input_img_h, input_img_w))

            mlvl_masks = []
            mlvl_positional_encodings = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))

            
            feats = self.transformer_encoder(mlvl_feats, mlvl_masks, mlvl_positional_encodings)
            
            split_size_or_sections = [None] * self.transformer_num_feature_levels

            for i in range(self.transformer_num_feature_levels):
                bs, _, h, w = mlvl_feats[i].shape
                split_size_or_sections[i] = h * w
                
            y = torch.split(feats, split_size_or_sections, dim=1)

            for i, z in enumerate(y):
                out.append(z.transpose(1, 2).view(bs, -1, mlvl_feats[i].size(2), mlvl_feats[i].size(3)))

            out = out[::-1]

        # NOTE: pixel-wise decoder to obtain the hr feature map
        multi_scale_features = []
        num_cur_levels = 0
        # append `out` with extra FPN levels (following MaskFormer, Mask2Former)
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.index[:self.num_fpn_levels][::-1]):

            x = inputs[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)

            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            y = output_conv(y)

            out.append(y)
        
        # the features in list out:
        # binsformer out = [1/32 enh feat, 1/16 enh feat, 1/8 enh feat, ... 
        #                   **DMSTransformer output(or naive inputs), low res to high res.**
        #                   **totally have self.transformer_num_feature_levels feats witch will interact with the bins queries**
        #                   temp feat, temp feat, temp feat, ..., per-pixel final-feat]

        for o in out:
            if num_cur_levels < self.transformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        
        # NOTE: transformer decoder
        per_pixel_feat = out[-1]
        pred_bins = []
        pred_depths = []
        pred_classes = []

        # deal with multi-scale feats
        mlvl_feats = multi_scale_features

        src = []
        pos = []
        size_list = []

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = mlvl_feats[0].size(2), mlvl_feats[0].size(3)
        img_masks = mlvl_feats[0].new_zeros(
            (batch_size, input_img_h, input_img_w))

        mlvl_masks = []
        for idx, feat in enumerate(mlvl_feats):
            size_list.append(feat.shape[-2:])
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                            size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            pos.append(
                self.positional_encoding(mlvl_masks[-1]).flatten(2) + self.level_embed.weight[idx][None, :, None])
            src.append(feat.flatten(2))

            # 4, 256, 14144 -> HW, N, C
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        
        multi_scale_infos = {'src':src, 'pos':pos, 'size_list':size_list}

        bs = per_pixel_feat.shape[0]
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pe = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_bins, predictions_logits, predictions_class = \
             self.transformer_decoder(multi_scale_infos, query_feat, query_pe, per_pixel_feat)

        # NOTE: depth estimation module
        self.norm = 'softmax'

        for item_bin, pred_logit, pred_class in \
            zip(predictions_bins, predictions_logits, predictions_class):
            
            if self.binsformer is False:
                pred_depth = F.relu(self.pred_depth(pred_logit)) + self.min_depth

            else:

                bins = item_bin.squeeze(dim=2)
                
                if self.norm == 'linear':
                    bins = torch.relu(bins)
                    eps = 0.1
                    bins = bins + eps
                elif self.norm == 'softmax':
                    bins = torch.softmax(bins, dim=1)
                else:
                    bins = torch.sigmoid(bins)
                bins = bins / bins.sum(dim=1, keepdim=True)

                bin_widths = (self.max_depth - self.min_depth) * bins  # .shape = N, dim_out
                bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
                bin_edges = torch.cumsum(bin_widths, dim=1)
                centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
                n, dout = centers.size()
                centers = centers.contiguous().view(n, dout, 1, 1)

                pred_logit = pred_logit.softmax(dim=1)
                pred_depth = torch.sum(pred_logit * centers, dim=1, keepdim=True)

                centers = self.hook_identify_center(centers)
                pred_logit = self.hook_identify_prob(pred_logit)
            
                pred_bins.append(bin_edges)
                pred_classes.append(pred_class) 

            pred_depths.append(pred_depth)

        return pred_depths, pred_bins, pred_classes

    def forward_train(self, img, inputs, img_metas, depth_gt, train_cfg, class_label=None):
        losses = dict()

        pred_depths, pred_bins, pred_classes = self.forward(inputs)

        aux_weight_dict = {}

        if train_cfg["aux_loss"]:

            for index, weight in zip(train_cfg["aux_index"], train_cfg["aux_weight"]):
                depth = pred_depths[index]

                depth = resize(
                    input=depth,
                    size=depth_gt.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                
                if self.binsformer is False:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight

                else:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight

                    if self.classify:
                        cls = pred_classes[index]
                        loss_ce, _ = self.loss_class(cls, class_label)
                        aux_weight_dict.update({'aux_loss_ce' + f"_{index}": loss_ce})

                    if self.with_loss_chamfer:
                        bin = pred_bins[index]
                        bins_loss = self.loss_chamfer(bin, depth_gt) * weight
                        aux_weight_dict.update({'aux_loss_chamfer' + f"_{index}": bins_loss})
                
                aux_weight_dict.update({'aux_loss_depth' + f"_{index}": depth_loss})
            
            losses.update(aux_weight_dict)

        # main loss
        depth = pred_depths[-1]
        
        depth = resize(
            input=depth,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)

        if self.binsformer is False:
            depth_loss = self.loss_decode(depth, depth_gt)
        else:
            depth_loss = self.loss_decode(depth, depth_gt)

            if self.classify:
                cls = pred_classes[-1]
                loss_ce, acc = self.loss_class(cls, class_label) 
                losses["loss_ce"] = loss_ce
                for index, topk in enumerate(acc):
                    losses["ce_acc_level_{}".format(index)] = topk

            if self.with_loss_chamfer:
                bin = pred_bins[-1]
                bins_loss = self.loss_chamfer(bin, depth_gt)
                losses["loss_chamfer"] = bins_loss

        losses["loss_depth"] = depth_loss

        log_imgs = self.log_images(img[0], pred_depths[0], depth_gt[0], img_metas[0])
        losses.update(**log_imgs)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):

        pred_depths, pred_bins, pred_classes = self.forward(inputs)

        return pred_depths[-1]