model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        out_indices=(2, 5, 8, 11),
        final_norm=False,
        with_cls_token=True,
        output_cls_token=True),
    decode_head=dict(
        type='DPTHead',
        in_channels=(768, 768, 768, 768),
        channels=256,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        readout_type='project',
        norm_cfg=None),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable