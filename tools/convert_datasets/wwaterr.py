# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Wwaterr dataset to mmsegmentation format')
    parser.add_argument('dataset_path', type=str, help='Wwaterr folder path')
    parser.add_argument('-f', '--fraction', type=float, default=0.8, help='fraction of train split')
    parser.add_argument('--size', type=int, default=1024, help='size of crop image')
    parser.add_argument('--overlap', type=int, default=512, help='overlap of crop image')
    parser.add_argument('-o', '--out_dir', type=str, default='data/Wwaterr', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    fraction = args.fraction
    size = args.size
    overlap = args.overlap
    out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    dataset_path_list = os.listdir(dataset_path)
    for name in ['images', 'masks']:
        assert name in dataset_path_list, f'{name} in not in {dataset_path}'

    # meta
    class_names = ['background', 'water']
    palette = [[0, 0, 0], [0, 0, 255]]
    print('Wwaterr dataset:')
    for cls, color in zip(class_names, palette):
        print(f'{cls} : {color}')

    image_list = os.listdir(osp.join(dataset_path, 'images'))
    num = len(image_list)
    num_train = int(num * fraction)
    num_val = num - num_train
    np.random.shuffle(image_list)
    train_list = image_list[:num_train]
    val_list = image_list[num_train:]

    # train
    cnt_train = 0
    for image in train_list:
        src_image_path = osp.join(dataset_path, 'images', image)
        src_mask_path = osp.join(dataset_path, 'masks', image)
        src_image = cv2.imread(src_image_path)
        src_mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
        dst_image = src_image
        _, dst_mask = cv2.threshold(src_mask, 128, 1, cv2.THRESH_BINARY)

        height, width, _ = src_image.shape
        if height <= size and width <= size:
            aspect_ratio = height / width
            if (height >= size // 2) and (width >= size // 2) and (aspect_ratio >= 0.5) and (aspect_ratio <= 2.0):
                dst_image_path = osp.join(out_dir, 'img_dir', 'train', f'{cnt_train:06d}.jpg')
                dst_mask_path = osp.join(out_dir, 'ann_dir', 'train', f'{cnt_train:06d}.png')
                print(f'copying {src_image_path} to {dst_image_path} ...')
                cv2.imwrite(dst_image_path, dst_image)
                print(f'copying {src_mask_path} to {dst_mask_path} ...')
                cv2.imwrite(dst_mask_path, dst_mask)
                cnt_train += 1
        else:
            for y in range(0, height, size - overlap):
                for x in range(0, width, size - overlap):
                    y_begin = y
                    y_end = y + size
                    if y_end > height:
                        y_begin = max(0, height - size)
                        y_end = height
                    x_begin = x
                    x_end = x + size
                    if x_end > width:
                        x_begin = max(0, width - size)
                        x_end = width
                    aspect_ratio = (y_end - y_begin + 1) / (x_end - x_begin + 1)
                    if (y_end - y_begin + 1 < size // 2) or (x_end - x_begin + 1 < size // 2) \
                            or (aspect_ratio < 0.5) or (aspect_ratio > 2.0):
                        continue
                    dst_image_path = osp.join(out_dir, 'img_dir', 'train', f'{cnt_train:06d}.jpg')
                    dst_mask_path = osp.join(out_dir, 'ann_dir', 'train', f'{cnt_train:06d}.png')
                    image_patch = dst_image[y_begin:y_end, x_begin:x_end, :]
                    mask_patch = dst_mask[y_begin:y_end, x_begin:x_end]
                    print(f'cropping {src_image_path} to {dst_image_path} ...')
                    cv2.imwrite(dst_image_path, image_patch)
                    print(f'cropping {src_mask_path} to {dst_mask_path} ...')
                    cv2.imwrite(dst_mask_path, mask_patch)
                    cnt_train += 1

    # val
    cnt_val = 0
    for image in val_list:
        src_image_path = osp.join(dataset_path, 'images', image)
        src_mask_path = osp.join(dataset_path, 'masks', image)
        src_image = cv2.imread(src_image_path)
        src_mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
        dst_image = src_image
        _, dst_mask = cv2.threshold(src_mask, 128, 1, cv2.THRESH_BINARY)

        height, width, _ = src_image.shape
        if height <= size and width <= size:
            aspect_ratio = height / width
            if (height >= size // 2) and (width >= size // 2) and (aspect_ratio >= 0.5) and (aspect_ratio <= 2.0):
                dst_image_path = osp.join(out_dir, 'img_dir', 'val', f'{cnt_val:06d}.jpg')
                dst_mask_path = osp.join(out_dir, 'ann_dir', 'val', f'{cnt_val:06d}.png')
                print(f'copying {src_image_path} to {dst_image_path} ...')
                cv2.imwrite(dst_image_path, dst_image)
                print(f'copying {src_mask_path} to {dst_mask_path} ...')
                cv2.imwrite(dst_mask_path, dst_mask)
                cnt_val += 1
        else:
            for y in range(0, height, size - overlap):
                for x in range(0, width, size - overlap):
                    y_begin = y
                    y_end = y + size
                    if y_end > height:
                        y_begin = max(0, height - size)
                        y_end = height
                    x_begin = x
                    x_end = x + size
                    if x_end > width:
                        x_begin = max(0, width - size)
                        x_end = width
                    aspect_ratio = (y_end - y_begin + 1) / (x_end - x_begin + 1)
                    if (y_end - y_begin + 1 < size // 2) or (x_end - x_begin + 1 < size // 2) \
                            or (aspect_ratio < 0.5) or (aspect_ratio > 2.0):
                        continue
                    dst_image_path = osp.join(out_dir, 'img_dir', 'val', f'{cnt_val:06d}.jpg')
                    dst_mask_path = osp.join(out_dir, 'ann_dir', 'val', f'{cnt_val:06d}.png')
                    image_patch = dst_image[y_begin:y_end, x_begin:x_end, :]
                    mask_patch = dst_mask[y_begin:y_end, x_begin:x_end]
                    print(f'cropping {src_image_path} to {dst_image_path} ...')
                    cv2.imwrite(dst_image_path, image_patch)
                    print(f'cropping {src_mask_path} to {dst_mask_path} ...')
                    cv2.imwrite(dst_mask_path, mask_patch)
                    cnt_val += 1

    print(f'train: {cnt_train}, val: {cnt_val}.')

    print('Done!')


if __name__ == '__main__':
    main()
