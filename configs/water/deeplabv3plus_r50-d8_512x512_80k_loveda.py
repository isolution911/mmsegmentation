_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=7),
    auxiliary_head=dict(num_classes=7))

work_dir = './work_dirs/water/deeplabv3plus_r50-d8_512x512_80k_loveda'
