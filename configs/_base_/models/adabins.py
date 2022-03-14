# model settings
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='EfficientNet'),
    decode_head=dict(
        type='AdabinsHead',
        in_channels=[24, 40, 64, 176, 2048],
        up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=128, # last one
        align_corners=True, # for upsample
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=10)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
