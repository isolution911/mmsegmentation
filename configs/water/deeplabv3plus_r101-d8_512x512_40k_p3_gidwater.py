_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/water.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet101_v1c')),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))
dataset_type = 'WaterDataset'
data_root = 'data/GID'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='dlp2t0.7/train'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='label2mask/val'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='label2mask/val'))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

work_dir = './work_dirs/water/deeplabv3plus_r101-d8_512x512_40k_p3_gidwater'
