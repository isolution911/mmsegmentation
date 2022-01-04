_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/water.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))
dataset_type = 'WaterDataset'
data_root = 'data/GID'
data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='dlp1t0.9/train'),
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

work_dir = './work_dirs/water/deeplabv3plus_r50-d8_512x512_40k_dlp1t0.9_gidwater'
