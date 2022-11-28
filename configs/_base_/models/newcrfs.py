# model settings
norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=norm_cfg,
        pretrain_style='official'),
    neck=dict(
        type='PSPNeck',
        in_channels=[192, 384, 768, 1536],
        channels=512),
    decode_head=dict(
        type='NewCRFHead',
        in_channels=[192, 384, 768, 1536],
        window_size=7,
        crf_dims=[128, 256, 512, 1024],
        v_dims=[64, 128, 256, 512],
        channels=128,
        min_depth=1e-3,
        max_depth=10),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable