# dataset settings
# We only use SUN RGB-D dataset for cross-dataset evaluation
dataset_type = 'SUNRGBDDataset'
data_root = 'data/SUNRGBD/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(0, 0),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=8000,
        split='SUNRGBD_val_splits.txt',
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=10))