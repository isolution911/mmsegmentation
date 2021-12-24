_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/wwaterr.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 3.0])),
    auxiliary_head=dict(
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[1.0, 2.0])))

work_dir = './work_dirs/water/deeplabv3plus_r50-d8_cbl_512x512_40k_wwaterr'
