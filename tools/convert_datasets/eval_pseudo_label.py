# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import mmcv
import cv2

from mmseg.apis import init_segmentor, generate_pseudo_label
from mmseg.core import intersect_and_union, pre_eval_to_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate the pseudo label')
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_ndwi_parser(subparsers)
    add_pseudo_parser(subparsers)
    args = parser.parse_args()
    return args


def add_ndwi_parser(subparsers):
    parser_ndwi = subparsers.add_parser('ndwi', help='evaluate ndwi2mask label')
    parser_ndwi.add_argument('img_dir', type=str, help='the directory of NirRGB images')
    parser_ndwi.add_argument('label_dir', type=str, help='the directory of labels')
    parser_ndwi.add_argument('--threshold', type=float, default=0.5, help='the threshold to generate ndwi2mask')
    parser_ndwi.set_defaults(func=ndwi)


def add_pseudo_parser(subparsers):
    parser_pseudo = subparsers.add_parser('pseudo', help='evaluate pseudo2mask label')
    parser_pseudo.add_argument('config', type=str, help='config file')
    parser_pseudo.add_argument('checkpoint', type=str, help='checkpoint file')
    parser_pseudo.add_argument('img_dir', type=str, help='the directory of images')
    parser_pseudo.add_argument('label_dir', type=str, help='the directory of labels')
    parser_pseudo.add_argument('--threshold', type=float, default=0.5, help='the threshold to generate pseudo label')
    parser_pseudo.add_argument('--device', type=str, default='cuda:9', help='device used for inference')
    parser_pseudo.set_defaults(func=pseudo)


def ndwi(args):
    img_dir = args.img_dir
    label_dir = args.label_dir
    threshold = args.threshold

    pre_eval_results = []
    img_list = os.listdir(img_dir)
    print('Evaluation start!')
    for i, image_name in enumerate(img_list):
        image_id = osp.splitext(image_name)[0]
        img_path = osp.join(img_dir, f'{image_id}.png')
        label_path = osp.join(label_dir, f'{image_id}.png')
        if i % 100 == 0:
            print(f'[{i}/{len(img_list)}]')
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        g, r, nir, b = cv2.split(img)
        np.seterr(divide='ignore', invalid='ignore')
        nir = nir.astype(float)
        g = g.astype(float)
        ndwi = (g - nir) / (g + nir)
        ndwi[ndwi == np.nan] = -1.0
        ndwi2mask = np.zeros_like(ndwi, dtype=np.uint8)
        ndwi2mask[ndwi >= threshold] = 1
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        pre_eval_results.append(
            intersect_and_union(ndwi2mask, label, 2, 255)
        )
    print('Done!')

    print(f'{args.task}: {args.threshold}')
    evaluate(pre_eval_results)


def pseudo(args):
    config = args.config
    checkpoint = args.checkpoint
    img_dir = args.img_dir
    label_dir = args.label_dir
    threshold = args.threshold
    device = args.device

    model = init_segmentor(config, checkpoint, device=device)
    pre_eval_results = []
    img_list = os.listdir(img_dir)
    print('Evaluation start!')
    for i, image_name in enumerate(img_list):
        image_id = osp.splitext(image_name)[0]
        img_path = osp.join(img_dir, f'{image_id}.jpg')
        label_path = osp.join(label_dir, f'{image_id}.png')
        if i % 100 == 0:
            print(f'[{i}/{len(img_list)}]')
        pseudo2mask = generate_pseudo_label(model, img_path, threshold=threshold)[0]
        pseudo2mask[pseudo2mask != 1] = 0
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        pre_eval_results.append(
            intersect_and_union(pseudo2mask, label, 2, 255)
        )
    print('Done!')

    print(f'{args.task}: {args.threshold}')
    evaluate(pre_eval_results)


def evaluate(pre_eval_results):
    metrics = ['mIoU', 'mDice', 'mFscore']
    ret_metrics = pre_eval_to_metrics(pre_eval_results, metrics)
    for key, value in ret_metrics.items():
        print(f'{key}: {value}')


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
