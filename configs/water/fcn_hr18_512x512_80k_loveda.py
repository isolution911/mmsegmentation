_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/loveda.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(decode_head=dict(num_classes=7))

work_dir = './work_dirs/water/fcn_hr18_512x512_80k_loveda'
