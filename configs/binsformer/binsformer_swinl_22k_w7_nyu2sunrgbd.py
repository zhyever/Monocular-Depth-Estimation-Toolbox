_base_ = [
    '../_base_/models/binsformer.py',
    '../_base_/default_runtime.py', 
    '../_base_/datasets/sun_rgbd.py', 
]

model = dict(
    pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth', # noqa
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    decode_head=dict(
        type='BinsFormerDecodeHead',
        class_num=25,
        in_channels=[192, 384, 768, 1536],
        conv_dim=512,
        min_depth=1e-3,
        max_depth=10,
        n_bins=64,
        index=[0, 1, 2, 3],
        trans_index=[1, 2, 3], # select index for cross-att
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=10),
        with_loss_chamfer=False, # do not use chamfer loss
        loss_chamfer=dict(type='BinsChamferLoss', loss_weight=1e-1),
        classify=True, # class embedding
        loss_class=dict(type='CrossEntropyLoss', loss_weight=1e-2),
        norm_cfg=dict(type='BN', requires_grad=True),
        transformer_encoder=dict( # default settings
            type='PureMSDEnTransformer',
            num_feature_levels=3,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', 
                        embed_dims=512, 
                        num_levels=3, 
                        num_points=8),
                    ffn_cfgs=dict(
                        embed_dims=512,
                        feedforward_channels=1024,
                        ffn_dropout=0.1,),
                    # feedforward_channels=1024,
                    # ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=256, normalize=True),
        transformer_decoder=dict(
            type='PixelTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            num_feature_levels=3,
            hidden_dim=512,
            operation='%',
            transformerlayers=dict(
                type='PixelTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=512,
                    num_heads=8,
                    dropout=0.0),
                ffn_cfgs=dict(
                    feedforward_channels=2048,
                    ffn_drop=0.0),
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')))),
    train_cfg=dict(
        aux_loss = True,
        aux_index = [2, 5],
        aux_weight = [1/4, 1/2]
    ),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
max_lr = 1e-4
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))
# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    warmup_iters=1600 * 8,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1600 * 24)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=1600, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"), 
                  less_keys=("abs_rel", "rmse"))

# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook') # TensorboardImageLoggerHook
    ])


find_unused_parameters=True