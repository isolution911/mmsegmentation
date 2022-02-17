_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/water.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        num_classes=2
    ),
    auxiliary_head=dict(in_channels=256,
                        channels=64,
                        num_classes=2))
dataset_type = 'WaterDataset'
data_root = 'data/GID'
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ndwi2mask/train'),
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

work_dir = './work_dirs/water/deeplabv3plus_r18-d8_512x512_40k_p1_gidwater'
