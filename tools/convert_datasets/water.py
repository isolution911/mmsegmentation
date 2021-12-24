# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Water dataset to mmsegmentation format')
    parser.add_argument('dataset_path', type=str, help='Water folder path')
    parser.add_argument('-f', '--fraction', type=float, default=0.8, help='fraction of train split')
    parser.add_argument('--size', type=int, default=1024, help='size of crop image')
    parser.add_argument('--overlap', type=int, default=256, help='overlap of crop image')
    parser.add_argument('-o', '--out_dir', type=str, default='data/Water', help='output path')
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
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'img_nirrgb_dir'))
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'img_nirrgb_dir', 'train'))
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'img_nirrgb_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    dataset_path_list = os.listdir(dataset_path)
    # for name in ['images_RGB', 'images_NirRGB', 'annotations']:
    #     assert name in dataset_path_list, f'{name} in not in {dataset_path}'
    for name in ['images_RGB', 'annotations']:
        assert name in dataset_path_list, f'{name} in not in {dataset_path}'

    # meta
    class_names = ['background', 'water']
    palette = [[0, 0, 0], [0, 0, 255]]
    print('Water dataset:')
    for cls, color in zip(class_names, palette):
        print(f'{cls} : {color}')

    image_list = os.listdir(osp.join(dataset_path, 'images_RGB'))
    num = len(image_list)
    num_train = int(num * fraction)
    num_val = num - num_train
    np.random.shuffle(image_list)
    train_list = image_list[:num_train]
    val_list = image_list[num_train:]
    print(f'train: {num_train}, val: {num_val}, which are both sampled from original dataset.')

    # train
    for image in train_list:
        image_id = osp.splitext(image)[0]
        src_image_path = osp.join(dataset_path, 'images_RGB', image)
        # src_image_nirrgb_path = osp.join(dataset_path, 'images_NirRGB', image)
        src_mask_path = osp.join(dataset_path, 'annotations', image)
        src_image = cv2.imread(src_image_path, -1)
        # src_image_nirrgb = cv2.imread(src_image_nirrgb_path, -1)
        src_mask = cv2.imread(src_mask_path, -1)

        height, width, _ = src_image.shape
        for y in range(0, height, size - overlap):
            for x in range(0, width, size - overlap):
                y_begin = y
                y_end = y + size
                if y_end > height:
                    y_begin = height - size
                    y_end = height
                x_begin = x
                x_end = x + size
                if x_end > width:
                    x_begin = width - size
                    x_end = width
                print(f'cropping patch ({y_begin}, {y_end}, {x_begin}, {x_end}) from {image_id} ...')
                image_patch = src_image[y_begin:y_end, x_begin:x_end, :]
                # image_nirrgb_patch = src_image_nirrgb[y_begin:y_end, x_begin:x_end, :]
                mask_patch = src_mask[y_begin:y_end, x_begin:x_end]
                patch_id = image_id + f'__{y_begin}_{y_end}_{x_begin}_{x_end}'
                dst_image_path = osp.join(out_dir, 'img_dir', 'train', patch_id + '.jpg')
                # dst_image_nirrgb_path = osp.join(out_dir, 'img_nirrgb_dir', 'train', patch_id + '.png')
                dst_label_path = osp.join(out_dir, 'ann_dir', 'train', patch_id + '.png')
                cv2.imwrite(dst_image_path, image_patch)
                # cv2.imwrite(dst_image_nirrgb_path, image_nirrgb_patch)
                cv2.imwrite(dst_label_path, mask_patch)

    # val
    for image in val_list:
        image_id = osp.splitext(image)[0]
        src_image_path = osp.join(dataset_path, 'images_RGB', image)
        # src_image_nirrgb_path = osp.join(dataset_path, 'images_NirRGB', image)
        src_mask_path = osp.join(dataset_path, 'annotations', image)
        src_image = cv2.imread(src_image_path, -1)
        # src_image_nirrgb = cv2.imread(src_image_nirrgb_path, -1)
        src_mask = cv2.imread(src_mask_path, -1)

        height, width, _ = src_image.shape
        for y in range(0, height, size - overlap):
            for x in range(0, width, size - overlap):
                y_begin = y
                y_end = y + size
                if y_end > height:
                    y_begin = height - size
                    y_end = height
                x_begin = x
                x_end = x + size
                if x_end > width:
                    x_begin = width - size
                    x_end = width
                print(f'cropping patch ({y_begin}, {y_end}, {x_begin}, {x_end}) from {image_id} ...')
                image_patch = src_image[y_begin:y_end, x_begin:x_end, :]
                # image_nirrgb_patch = src_image_nirrgb[y_begin:y_end, x_begin:x_end, :]
                mask_patch = src_mask[y_begin:y_end, x_begin:x_end]
                patch_id = image_id + f'__{y_begin}_{y_end}_{x_begin}_{x_end}'
                dst_image_path = osp.join(out_dir, 'img_dir', 'val', patch_id + '.jpg')
                # dst_image_nirrgb_path = osp.join(out_dir, 'img_nirrgb_dir', 'val', patch_id + '.png')
                dst_label_path = osp.join(out_dir, 'ann_dir', 'val', patch_id + '.png')
                cv2.imwrite(dst_image_path, image_patch)
                # cv2.imwrite(dst_image_nirrgb_path, image_nirrgb_patch)
                cv2.imwrite(dst_label_path, mask_patch)

    print('Done!')


if __name__ == '__main__':
    main()
