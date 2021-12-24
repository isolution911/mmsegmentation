# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pandas as pd
import shutil
import numpy as np
import mmcv
import cv2
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DeepGlobe dataset to mmsegmentation format')
    parser.add_argument('dataset_path', type=str, help='DeepGlobe folder path')
    parser.add_argument('-f', '--fraction', type=float, default=0.8, help='fraction of train split')
    parser.add_argument('-o', '--out_dir', type=str, default='data/DeepGlobe', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    fraction = args.fraction
    out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'extra'))

    dataset_path_list = os.listdir(dataset_path)
    for name in ['train', 'valid', 'test', 'metadata.csv', 'class_dict.csv']:
        assert name in dataset_path_list, f'{name} in not in {dataset_path}'

    # meta
    shutil.copy(osp.join(dataset_path, 'metadata.csv'), osp.join(out_dir, 'extra', 'dataset.csv'))
    meta_data = pd.read_csv(osp.join(dataset_path, 'metadata.csv'))

    # class
    shutil.copy(osp.join(dataset_path, 'class_dict.csv'), osp.join(out_dir, 'extra', 'class.csv'))
    class_info = pd.read_csv(osp.join(dataset_path, 'class_dict.csv'))
    class_names = class_info['name'].tolist()
    palette = class_info[['r', 'g', 'b']].values.tolist()
    print('DeepGlobe dataset:')
    for cls, color in zip(class_names, palette):
        print(f'{cls} : {color}')

    # train + val
    meta_train = meta_data[meta_data['split'] == 'train']
    meta_training = meta_train.sample(frac=fraction, random_state=42)
    meta_validation = meta_train.drop(meta_training.index)

    print(f'train: {len(meta_training)}, val: {len(meta_validation)}, which are both sampled from original train.')

    for _, row in meta_training.iterrows():
        img_id = row['image_id']
        # img
        src_img_path = osp.join(dataset_path, row['sat_image_path'])
        dst_img_path = osp.join(out_dir, 'img_dir', 'train', f'{img_id}.jpg')
        print(f'copying {src_img_path} to {dst_img_path} ...')
        shutil.copy(src_img_path, dst_img_path)
        # mask
        src_mask_path = osp.join(dataset_path, row['mask_path'])
        dst_mask_path = osp.join(out_dir, 'ann_dir', 'train', f'{img_id}.png')
        print(f'converting {src_mask_path} to {dst_mask_path} ...')
        src_mask = Image.open(src_mask_path)
        src_mask_numpy = np.asarray(src_mask)
        for i in range(src_mask_numpy.ndim):
            _, src_mask_numpy[:, :, i] = cv2.threshold(src_mask_numpy[:, :, i], 128, 255, cv2.THRESH_BINARY)
        src_mask_id = src_mask_numpy.dot(np.array([[1], [256], [65536]])).squeeze(-1)
        dst_mask_numpy = 255 * np.ones_like(src_mask_id)
        for i, color in enumerate(palette):
            color_id = color[0] + color[1] * 256 + color[2] * 65536
            dst_mask_numpy[src_mask_id == color_id] = i

        dst_mask = Image.fromarray(dst_mask_numpy.astype(np.uint8))
        dst_mask.save(dst_mask_path)

    for _, row in meta_validation.iterrows():
        img_id = row['image_id']
        # img
        src_img_path = osp.join(dataset_path, row['sat_image_path'])
        dst_img_path = osp.join(out_dir, 'img_dir', 'val', f'{img_id}.jpg')
        print(f'copying {src_img_path} to {dst_img_path} ...')
        shutil.copy(src_img_path, dst_img_path)
        # mask
        src_mask_path = osp.join(dataset_path, row['mask_path'])
        dst_mask_path = osp.join(out_dir, 'ann_dir', 'val', f'{img_id}.png')
        print(f'converting {src_mask_path} to {dst_mask_path} ...')
        src_mask = Image.open(src_mask_path)
        src_mask_numpy = np.asarray(src_mask)
        for i in range(src_mask_numpy.ndim):
            _, src_mask_numpy[:, :, i] = cv2.threshold(src_mask_numpy[:, :, i], 128, 255, cv2.THRESH_BINARY)
        src_mask_id = src_mask_numpy.dot(np.array([[1], [256], [65536]])).squeeze(-1)
        dst_mask_numpy = 255 * np.ones_like(src_mask_id)
        for i, color in enumerate(palette):
            color_id = color[0] + color[1] * 256 + color[2] * 65536
            dst_mask_numpy[src_mask_id == color_id] = i

        dst_mask = Image.fromarray(dst_mask_numpy.astype(np.uint8))
        dst_mask.save(dst_mask_path)

    # test
    # meta_valid = meta_data[meta_data['split'] == 'valid']
    # meta_test = meta_data[meta_data['split'] == 'test']
    meta_testing = meta_data[meta_data['split'] != 'train']

    print(f'test: {len(meta_testing)}, which is from original valid and test.')

    for _, row in meta_testing.iterrows():
        img_id = row['image_id']
        # img
        src_img_path = osp.join(dataset_path, row['sat_image_path'])
        dst_img_path = osp.join(out_dir, 'img_dir', 'test', f'{img_id}.jpg')
        print(f'copying {src_img_path} to {dst_img_path} ...')
        shutil.copy(src_img_path, dst_img_path)

    meta_training = meta_training.reset_index(drop=True)
    meta_validation = meta_validation.reset_index(drop=True)
    meta_testing = meta_testing.reset_index(drop=True)
    meta_training.to_csv(osp.join(out_dir, 'extra', 'train.csv'), index=False)
    meta_validation.to_csv(osp.join(out_dir, 'extra', 'val.csv'), index=False)
    meta_testing.to_csv(osp.join(out_dir, 'extra', 'test.csv'), index=False)

    print('Done!')


if __name__ == '__main__':
    main()
