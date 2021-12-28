# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import cv2
from PIL import Image
from libtiff import TIFF


def main():
    # # opencv
    # image_rgb = cv2.imread('demo/images/demo_rgb.tif', cv2.IMREAD_UNCHANGED)
    # cv2.imwrite('demo/images/demo_rgb2rgb.jpg', image_rgb)
    #
    # image_nirrgb = cv2.imread('demo/images/demo_nirrgb.tif', cv2.IMREAD_UNCHANGED)
    # g, r, nir, b = cv2.split(image_nirrgb.astype(float))
    #
    # np.seterr(divide='ignore', invalid='ignore')
    # g = g * (255.0 / b)
    # r = r * (255.0 / b)
    # nir = nir * (255.0 / b)
    # g[g == np.nan] = 0.0
    # r[r == np.nan] = 0.0
    # nir[nir == np.nan] = 0.0
    #
    # ndwi = (g - nir) / (g + nir)
    # ndwi[ndwi == np.nan] = -1.0
    # ndwi_gray = (((ndwi + 1.0) / 2.0) * 255).astype(np.uint8)
    # cv2.imwrite('demo/images/ndwi_gray.png', ndwi_gray)
    #
    # # _, ndwi_binary = cv2.threshold(ndwi_gray, 192, 255, cv2.THRESH_BINARY)
    # ndwi_binary = np.zeros_like(ndwi, dtype=np.uint8)
    # threshold = 0.5
    # ndwi_binary[ndwi >= threshold] = 255
    # cv2.imwrite('demo/images/ndwi_binary.png', ndwi_binary)
    #
    # g = g.astype(np.uint8)
    # r = r.astype(np.uint8)
    # nir = nir.astype(np.uint8)
    # b = b.astype(np.uint8)
    # image_nirrgb = cv2.merge([g, r, nir, b])
    # cv2.imwrite('demo/images/demo_nirrgb2nirrgb.png', image_nirrgb)
    # image = cv2.merge([b, g, r])
    # cv2.imwrite('demo/images/demo_nirrgb2rgb.jpg', image)

    # # pillow
    # image_rgb = Image.open('demo/images/demo_rgb.tif')
    # image_rgb.save('demo/images/demo_rgb2rgb.jpg')
    #
    # image_nirrgb = Image.open('demo/images/demo_nirrgb.tif')
    # image_nirrgb.save('demo/images/demo_nirrgb2nirrgb.png')
    # image_nirrgb = np.array(image_nirrgb)
    # nir, r, g, b = cv2.split(image_nirrgb)
    # image = cv2.merge([r, g, b])
    # image = Image.fromarray(image).convert('RGB')
    # image.save('demo/images/demo_nirrgb2rgb.jpg')
    #
    # np.seterr(divide='ignore', invalid='ignore')
    # g = g.astype(float)
    # nir = nir.astype(float)
    # ndwi = (g - nir) / (g + nir)
    # ndwi[ndwi == np.nan] = -1.0
    # ndwi_gray = (((ndwi + 1.0) / 2.0) * 255).astype(np.uint8)
    # ndwi_gray = Image.fromarray(ndwi_gray).convert('L')
    # ndwi_gray.save('demo/images/ndwi_gray.png')
    # ndwi_binary = np.zeros_like(ndwi, dtype=np.uint8)
    # threshold = 0.5
    # ndwi_binary[ndwi >= threshold] = 255
    # ndwi_binary = Image.fromarray(ndwi_binary).convert('L')
    # ndwi_binary.save('demo/images/ndwi_binary.png')
    #
    # image_mask = Image.open('demo/images/demo_mask.tif')
    # image_mask = np.array(image_mask)
    # image_mask = image_mask[:, :, 0] * 1 + image_mask[:, :, 1] * 256 + image_mask[:, :, 2] * 65536
    # color_water = 0 * 1 + 0 * 256 + 255 * 65536
    # mask = np.zeros_like(image_mask, dtype=np.uint8)
    # mask[image_mask == color_water] = 255
    # mask = Image.fromarray(mask).convert('L')
    # mask.save('demo/images/mask_water.png')
    #
    # ndwi_binary = np.array(ndwi_binary).astype(np.int64)
    # mask = np.array(mask).astype(np.int64)
    # image_diff = np.zeros_like(mask)
    # image_diff[ndwi_binary + mask == 255 * 2] = color_water
    # image_diff[ndwi_binary - mask == 255] = 255 * 1 + 0 * 256 + 0 * 65536
    # image_diff[ndwi_binary - mask == -255] = 0 * 1 + 255 * 256 + 0 * 65536
    # image_r = image_diff % 256
    # image_g = (image_diff // 256) % 256
    # image_b = image_diff // 65536
    # image_comp = cv2.merge([image_r, image_g, image_b]).astype(np.uint8)
    # image_comp = Image.fromarray(image_comp).convert('RGB')
    # image_comp.save('demo/images/diff.jpg')

    # libtiff
    tif_rgb = TIFF.open('demo/images/demo_rgb.tif', mode='r')
    image_rgb = tif_rgb.read_image()
    r, g, b = cv2.split(image_rgb)
    image_rgb2rgb = cv2.merge([b, g, r])
    cv2.imwrite('demo/images/demo_rgb2rgb.jpg', image_rgb2rgb)

    tif_nirrgb = TIFF.open('demo/images/demo_nirrgb.tif', mode='r')
    image_nirrgb = tif_nirrgb.read_image()
    nir, r, g, b = cv2.split(image_nirrgb)
    image_nirrgb2nirrgb = cv2.merge([g, r, nir, b])
    cv2.imwrite('demo/images/demo_nirrgb2nirrgb.png', image_nirrgb2nirrgb)
    image_nirrgb2rgb = cv2.merge([b, g, r])
    cv2.imwrite('demo/images/demo_nirrgb2rgb.jpg', image_nirrgb2rgb)

    np.seterr(divide='ignore', invalid='ignore')
    g = g.astype(float)
    nir = nir.astype(float)

    ndwi = (g - nir) / (g + nir)
    ndwi[ndwi == np.nan] = -1.0
    ndwi_gray = (((ndwi + 1.0) / 2.0) * 255).astype(np.uint8)
    cv2.imwrite('demo/images/ndwi_gray.png', ndwi_gray)

    # _, ndwi_binary = cv2.threshold(ndwi_gray, 192, 255, cv2.THRESH_BINARY)
    ndwi_binary = np.zeros_like(ndwi, dtype=np.uint8)
    threshold = 0.5
    ndwi_binary[ndwi >= threshold] = 255
    cv2.imwrite('demo/images/ndwi_binary.png', ndwi_binary)


if __name__ == '__main__':
    main()
