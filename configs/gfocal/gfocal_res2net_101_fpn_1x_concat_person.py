_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    pretrained='/data-nas/sy/model_zoo/pretrained_model/res2net101_v1d_26w_4s_mmdetv2-f0a600f9.pth',
    backbone=dict(
        type='Res2Net', depth=101, scales=4, base_width=26),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# data
dataset_type = 'CocoDataset'
data_root = '/data-nas/sy/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # train=[  # concat coco & objects365 person
    #     dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'coco/annotations/instances_train2017_person.json',
    #         img_prefix=data_root + 'coco/train2017/',
    #         pipeline=train_pipeline),
    #     dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'objects365/annotations/objects365_train_fixed_sel_human.json',
    #         img_prefix=data_root + 'objects365/train/',
    #         pipeline=train_pipeline)
    # ],
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'objects365/annotations/objects365_train_fixed_sel_human.json',
        img_prefix=data_root + 'objects365/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'objects365/annotations/objects365_val_fixed_sel_human.json',
        img_prefix=data_root + 'objects365/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'objects365/annotations/objects365_val_fixed_sel_human.json',
        img_prefix=data_root + 'objects365/val/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# load_from='/data-nas/sy/model_zoo/mmdet/gfl_v2/gfocal_res2net_101_fpn_1x_coco/epoch_12.pth'
