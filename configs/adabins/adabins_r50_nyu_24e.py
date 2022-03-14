_base_ = [
    '../_base_/models/adabins.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        style='pytorch',
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50'),),
    decode_head=dict(
        in_channels=[64, 256, 512, 1024, 2048],
        up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=128, # last one
        min_depth=1e-3,
        max_depth=10,
        norm_cfg=norm_cfg),
    )

find_unused_parameters=True
SyncBN=True

# runtime
evaluation = dict(interval=1)