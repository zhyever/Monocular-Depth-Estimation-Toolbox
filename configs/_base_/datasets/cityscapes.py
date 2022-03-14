# dataset settings
dataset_type = 'CSDataset'
data_root = 'data/cityscapes'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (352, 704)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DisparityLoadAnnotations'),
    dict(type='Resize', img_scale=(1216, 352), keep_ratio=False),
    dict(type='KBCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 704)),
    dict(type='ColorAug', prob=1, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1216, 352), keep_ratio=False),
    dict(type='KBCrop', depth=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 2048),
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
    train=dict(
        type=dataset_type,
        data_root='data/cityscapesExtra',
        img_dir='leftImg8bit',
        cam_dir='camera',
        ann_dir='disparity',
        depth_scale=256,
        split='cityscapes_train.txt',
        pipeline=train_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=200),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_trainvaltest',
        cam_dir='camera',
        ann_dir='disparity_trainvaltest',
        depth_scale=256,
        split='cityscapes_val.txt',
        pipeline=test_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=200),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit_trainvaltest',
        cam_dir='camera',
        ann_dir='disparity_trainvaltest',
        depth_scale=256,
        split='cityscapes_val.txt',
        pipeline=test_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=200))

