# Copyright (c) OpenMMLab. All rights reserved.
import torch

import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

from depth.ops import resize
from depth.models.builder import NECKS

@NECKS.register_module()
class HAHIHeteroNeck(BaseModule):
    """HAHIHeteroNeck.

        HAHI in `DepthFormer <https://arxiv.org/abs/2203.14211>`_. For heterogenenious cnn- and transformer- features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels.
        embedding_dim (int): Feature dimension in HAHI.
        positional_encoding (dict): position encoding used in attention modules
        scales (List[float]): Scale factors for each input feature map.
            Default: [1, 1, 1, 1]
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer in ConvModule. Default: dict(type='ReLU', inplace=True).
        cross_att (bool): Whether to apply cross attention in HAHI. Default: True.
        self_att (bool): Whether to apply self attention in HAHI. Default: True.
        num_points (int): The number of reference points used in attention modules. Default: 8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 embedding_dim,
                 positional_encoding,
                 scales=[1, 1, 1, 1],
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 cross_att=True,
                 self_att=True,
                 num_points=8):
        super(HAHIHeteroNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.cross_att = cross_att
        self.self_att = self_att
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.embedding_dim = embedding_dim
        
        self.lateral_convs = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.lateral_convs.append(
                ConvModule(in_channel,
                           out_channel,
                           kernel_size=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))

        self.trans_proj = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels[1:], out_channels[1:]):
            self.trans_proj.append(
                ConvModule(out_channel,
                           self.embedding_dim,
                           kernel_size=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))

        self.trans_fusion = nn.ModuleList()
        for in_channel, out_channel in zip(out_channels[1:], out_channels[1:]):
            self.trans_fusion.append(
                ConvModule(out_channel + self.embedding_dim,
                           out_channel,
                           kernel_size=3,
                           padding=1,
                           stride=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))

        self.conv_proj = nn.Sequential(
            ConvModule(in_channels[0],
                       self.embedding_dim,
                       kernel_size=1,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg))
        
        self.conv_fusion = nn.Sequential(
            ConvModule(in_channels[0] + self.embedding_dim,
                       out_channels[0],
                       kernel_size=3,
                       padding=1,
                       stride=1,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg))

        ########################################

        num_feature_levels = 4 # transformer feature level

        self.trans_positional_encoding = build_positional_encoding(positional_encoding)
        self.conv_positional_encoding = build_positional_encoding(positional_encoding)

        self.reference_points = nn.Linear(self.embedding_dim, 2)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, self.embedding_dim))

        self.multi_att = MultiScaleDeformableAttention(embed_dims=self.embedding_dim,
                                                       num_levels=4,
                                                       num_heads=8,
                                                       num_points=num_points,
                                                       batch_first=True)
        self.self_attn = MultiScaleDeformableAttention(embed_dims=self.embedding_dim,
                                                       num_levels=4,
                                                       num_heads=8,
                                                       num_points=num_points,
                                                       batch_first=True)
        
    # init weight
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
            
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # input projection
        feats_projed = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        feats_trans = feats_projed[1:]
        feat_conv = feats_projed[0]

        # HI (deformable self attention)
        masks = []
        src_flattens = []
        mask_flatten = []
        spatial_shapes = []
        lvl_pos_embed_flatten = []
        for i in range(len(feats_trans)):
            bs, c, h, w = feats_trans[i].shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            mask = torch.zeros_like(feats_trans[i][:, 0, :, :]).type(torch.bool)
            masks.append(mask)
            # pos = self.trans_positional_encoding(feats_trans[i], mask)
            pos = self.trans_positional_encoding(mask)
            mask = mask.flatten(1)
            pos_embed = pos.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[i].view(1, 1, -1)

            feat = self.trans_proj[i](feats_trans[i])
            flatten_feat = feat.flatten(2).transpose(1, 2)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            src_flattens.append(flatten_feat)

        src_flatten = torch.cat(src_flattens, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
        if self.self_att:
            src = self.self_attn(
                src_flatten,
                key=None,
                value=None,
                identity=None,
                query_pos=lvl_pos_embed_flatten,
                key_padding_mask=None,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,)
        else:
            src = src_flatten

        # HA (deformable cross attention)
        conv_skip = self.conv_proj(feat_conv)
        bs, c, h, w = conv_skip.shape
        query_mask = torch.zeros_like(conv_skip[:, 0, :, :]).type(torch.bool)
        query = conv_skip.flatten(2).transpose(1, 2)
        # query_embed = self.conv_positional_encoding(conv_skip, query_mask).flatten(2).transpose(1, 2)
        query_embed = self.conv_positional_encoding(query_mask).flatten(2).transpose(1, 2)
        reference_points = self.reference_points(query_embed).sigmoid()
        reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

        if self.cross_att:
            fusion_res_conv = self.multi_att(
                query,
                key=None,
                value=src,
                identity=None,
                query_pos=query_embed,
                key_padding_mask=None,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,)
        else:
            fusion_res_conv = query

        fusion_res_conv = fusion_res_conv.permute(0, 2, 1).reshape(bs, c, h, w)
        fusion_res_conv = self.conv_fusion(torch.cat([fusion_res_conv, feat_conv], dim=1))


        # unfold the feats back to the origin shape
        start = 0
        fusion_res_trans = []
        for i in range(len(feats_trans)):
            bs, c, h, w = feats_trans[i].shape
            end = start + h * w
            feat = src[:, start:end, :].permute(0, 2, 1).contiguous()
            start = end
            feat = feat.reshape(bs, self.embedding_dim, h, w)
            fusion_res_trans.append(torch.cat([feats_trans[i], feat], dim=1))

        # fusion 3x3 conv
        outs = []
        for i in range(len(feats_trans)):
            if self.scales[i] != 1:
                x_resize = resize(
                    fusion_res_trans[i], scale_factor=self.scales[i], mode='bilinear')
            else:
                x_resize = fusion_res_trans[i]
            x_resize = self.trans_fusion[i](x_resize)
            outs.append(x_resize)
        outs.insert(0, fusion_res_conv)

        return tuple(outs)

