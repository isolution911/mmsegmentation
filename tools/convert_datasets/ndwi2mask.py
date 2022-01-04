# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import cv2
from libtiff import TIFF


def parse_args():
    parser = argparse.ArgumentParser(description='Convert NDWI computed from NirRGB to mask')
    parser.add_argument('-i', '--in_dir', type=str, help='input path')
    parser.add_argument('-o', '--out_dir', type=str, help='output path')
    parser.add_argument('--threshold_low', type=float, default=0.3, help='low threshold to convert ndwi2mask')
    parser.add_argument('--threshold_high', type=float, default=0.7, help='high threshold to convert ndwi2mask')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    low = args.threshold_low
    high = args.threshold_high

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'val'))

    for split in ['train', 'val']:
        image_list = os.listdir(osp.join(in_dir, split))
        print(f'Converting {len(image_list)} NirRGB images of {split} split to ndwi2mask.')

        for image_name in image_list:
            image_id = osp.splitext(image_name)[0]
            in_path = osp.join(in_dir, split, image_name)
            out_path = osp.join(out_dir, split, f'{image_id}.png')
            print(f'converting {in_path} to {out_path} ...')
            # # load .tif
            # tif_nirrgb = TIFF.open(in_path, mode='r')
            # image_nirrgb = tif_nirrgb.read_image()
            # nir, r, g, b = cv2.split(image_nirrgb)
            # load .png
            image_nirrgb = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            g, r, nir, b = cv2.split(image_nirrgb)
            np.seterr(divide='ignore', invalid='ignore')
            nir = nir.astype(float)
            g = g.astype(float)
            ndwi = (g - nir) / (g + nir)
            ndwi[ndwi == np.nan] = -1.0
            ndwi2mask = 255 * np.ones_like(ndwi, dtype=np.uint8)
            ndwi2mask[ndwi <= low] = 0
            ndwi2mask[ndwi >= high] = 1
            cv2.imwrite(out_path, ndwi2mask)

    print('Done!')


if __name__ == '__main__':
    main()
