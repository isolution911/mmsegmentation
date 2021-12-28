# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import cv2
from libtiff import TIFF


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GID dataset to mmsegmentation format')
    parser.add_argument('dataset_path', type=str, help='GID folder path')
    parser.add_argument('-f', '--fraction', type=float, default=0.8, help='fraction of train split')
    parser.add_argument('--size', type=int, default=1024, help='size of crop image')
    parser.add_argument('--overlap', type=int, default=256, help='overlap of crop image')
    parser.add_argument('--threshold_low', type=float, default=0.3, help='low threshold to convert ndwi2mask')
    parser.add_argument('--threshold_high', type=float, default=0.7, help='high threshold to convert ndwi2mask')
    parser.add_argument('-o', '--out_dir', type=str, default='data/GID', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    fraction = args.fraction
    size = args.size
    overlap = args.overlap
    low = args.threshold_low
    high = args.threshold_high
    out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_nirrgb_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_nirrgb_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_nirrgb_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'label2mask'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'label2mask', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'label2mask', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ndwi2mask'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ndwi2mask', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ndwi2mask', 'val'))

    dataset_path_list = os.listdir(dataset_path)
    for name in ['image_RGB', 'image_NirRGB', 'label_5classes']:
        assert name in dataset_path_list, f'{name} in not in {dataset_path}'

    # meta
    class_names = ['built_up', 'farmland', 'forest', 'meadow', 'water', 'unknown']
    palette = [[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255], [0, 0, 0]]
    print('GID dataset:')
    for cls, color in zip(class_names, palette):
        print(f'{cls} : {color}')

    image_list = os.listdir(osp.join(dataset_path, 'image_RGB'))
    num = len(image_list)
    num_train = int(num * fraction)
    num_val = num - num_train
    np.random.shuffle(image_list)
    train_list = image_list[:num_train]
    val_list = image_list[num_train:]
    print(f'train: {num_train}, val: {num_val}, which are both sampled from original dataset.')

    for split, split_list in zip(['train', 'val'], [train_list, val_list]):
        for image_name in split_list:
            image_id = osp.splitext(image_name)[0]
            src_image_path = osp.join(dataset_path, 'image_RGB', image_name)
            src_image_nirrgb_path = osp.join(dataset_path, 'image_NirRGB', image_name)
            src_mask_path = osp.join(dataset_path, 'label_5classes', f'{image_id}_label.tif')
            # rgb
            src_tif = TIFF.open(src_image_path, mode='r')
            src_image = src_tif.read_image()
            src_image = src_image[:, :, [2, 1, 0]]
            # src_image = cv2.imread(src_image_path, -1)
            # nirrgb
            src_tif_nirrgb = TIFF.open(src_image_nirrgb_path, mode='r')
            src_image_nirrgb = src_tif_nirrgb.read_image()
            src_image_nirrgb = src_image_nirrgb[:, :, [2, 1, 0, 3]]
            # src_image_nirrgb = cv2.imread(src_image_nirrgb_path, -1)
            # mask
            print(f'converting {src_mask_path} to label ...')
            src_tif_mask = TIFF.open(src_mask_path, mode='r')
            src_mask = src_tif_mask.read_image()
            # src_mask = src_mask[:, :, [2, 1, 0]]
            # src_mask = cv2.imread(src_mask_path, -1)
            for i in range(src_mask.ndim):
                _, src_mask[:, :, i] = cv2.threshold(src_mask[:, :, i], 128, 255, cv2.THRESH_BINARY)
            src_mask_id = src_mask.astype(np.int32).dot(np.array([[1], [256], [65536]], dtype=np.int32)).squeeze(-1)
            # label
            src_label = 255 * np.ones_like(src_mask_id, dtype=np.uint8)
            for i, color in enumerate(palette):
                color_id = color[0] + color[1] * 256 + color[2] * 65536
                src_label[src_mask_id == color_id] = i
            # label2mask
            src_label2mask = np.zeros_like(src_mask_id, dtype=np.uint8)
            color_water = 0 * 1 + 0 * 256 + 255 * 65536
            src_label2mask[src_mask_id == color_water] = 1
            # ndwi2mask
            g, r, nir, b = cv2.split(src_image_nirrgb)
            np.seterr(divide='ignore', invalid='ignore')
            nir = nir.astype(float)
            g = g.astype(float)
            ndwi = (g - nir) / (g + nir)
            ndwi[ndwi == np.nan] = -1.0
            src_ndwi2mask = 255 * np.ones_like(ndwi, dtype=np.uint8)
            src_ndwi2mask[ndwi <= low] = 0
            src_ndwi2mask[ndwi >= high] = 1

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
                    image_nirrgb_patch = src_image_nirrgb[y_begin:y_end, x_begin:x_end, :]
                    label_patch = src_label[y_begin:y_end, x_begin:x_end]
                    label2mask_patch = src_label2mask[y_begin:y_end, x_begin:x_end]
                    ndwi2mask_patch = src_ndwi2mask[y_begin:y_end, x_begin:x_end]
                    patch_id = image_id + f'__{y_begin}_{y_end}_{x_begin}_{x_end}'
                    dst_image_path = osp.join(out_dir, 'img_dir', split, patch_id + '.jpg')
                    dst_image_nirrgb_path = osp.join(out_dir, 'img_nirrgb_dir', split, patch_id + '.png')
                    dst_label_path = osp.join(out_dir, 'ann_dir', split, patch_id + '.png')
                    dst_label2mask_path = osp.join(out_dir, 'label2mask', split, patch_id + '.png')
                    dst_ndwi2mask_path = osp.join(out_dir, 'ndwi2mask', split, patch_id + '.png')
                    cv2.imwrite(dst_image_path, image_patch)
                    cv2.imwrite(dst_image_nirrgb_path, image_nirrgb_patch)
                    cv2.imwrite(dst_label_path, label_patch)
                    cv2.imwrite(dst_label2mask_path, label2mask_patch)
                    cv2.imwrite(dst_ndwi2mask_path, ndwi2mask_patch)

    print('Done!')


if __name__ == '__main__':
    main()
