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

work_dir = './work_dirs/water/self_training/deeplabv3plus_r101-d8_512x512_40k_p3_gidwater'
