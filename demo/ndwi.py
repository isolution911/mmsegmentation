# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import cv2


def main():
    image_rgb = cv2.imread('demo/demo_rgb.tif', -1)
    cv2.imwrite('demo/patches/demo_rgb.jpg', image_rgb)

    image_nirrgb = cv2.imread('demo/demo_nirrgb.tif', -1)
    cv2.imwrite('demo/patches/demo_nirrgb.png', image_nirrgb)
    g, r, nir, b = cv2.split(image_nirrgb)
    image = cv2.merge([r, g, b])
    cv2.imwrite('demo/patches/demo_nirrgb.jpg', image)

    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (g - nir) / (g + nir)
    ndwi[ndwi == np.nan] = -1.0
    ndwi_gray = np.uint8((ndwi + 1.0) * 255 / 2.0)
    cv2.imwrite('demo/patches/ndwi_gray.png', ndwi_gray)
    # _, ndwi_binary = cv2.threshold(ndwi_gray, 196, 255, cv2.THRESH_BINARY)
    ndwi_binary = np.zeros_like(ndwi, dtype=np.uint8)
    ndwi_binary[ndwi <= 0.999] = 255
    cv2.imwrite('demo/patches/ndwi_binary.png', ndwi_binary)


if __name__ == '__main__':
    main()
