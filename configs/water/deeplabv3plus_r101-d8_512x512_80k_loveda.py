_base_ = './deeplabv3plus_r50-d8_512x512_80k_loveda.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnet101_v1c')))

work_dir = './work_dirs/water/deeplabv3plus_r101-d8_512x512_80k_loveda'
