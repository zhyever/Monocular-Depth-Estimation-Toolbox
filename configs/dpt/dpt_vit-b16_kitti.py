_base_ = [
    '../_base_/models/dpt.py', '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0),
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
max_lr=1e-4
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2, workers_per_gpu=2)

evaluation = dict(interval=1)