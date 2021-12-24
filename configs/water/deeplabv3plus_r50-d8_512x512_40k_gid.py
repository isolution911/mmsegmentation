_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/gid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6))

work_dir = './work_dirs/water/deeplabv3plus_r50-d8_512x512_40k_gid'
