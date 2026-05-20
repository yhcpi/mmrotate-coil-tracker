_base_ = ['../_base_/default_runtime.py']

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

angle_version = 'le90'

model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=1,
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=False,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1,
            loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=200,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=50))

data_root = 'data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PolyRandomRotate', rotate_ratio=0.5, mode='range', angles_range=10, auto_bound=True),
    dict(type='RResize', img_scale=(1600, 900)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 900),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RailDataset',
        ann_file=data_root + 'train/labels/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type='RailDataset',
        ann_file=data_root + 'val/labels/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type='RailDataset',
        ann_file=data_root + 'test/labels/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version=angle_version))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1.0 / 3,
    min_lr=1e-6)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP')
