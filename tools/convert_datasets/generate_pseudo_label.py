# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import mmcv
import cv2
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, generate_pseudo_label
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser(prog='generate_pseudo_label', usage='pseudo label generation',
                            description='MMSegmentation pseudo label generation')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('-i', '--input_dir', help='Directory of input image(s)')
    parser.add_argument('-o', '--output_dir', help='Directory of output pseudo label(s)')
    parser.add_argument('-t', '--threshold', type=float, default=0.9,
                        help='Threshold to generate pseudo label. In [0, 1] range.')
    parser.add_argument('--device', default='cuda:9', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # train and val
    mmcv.mkdir_or_exist(args.output_dir)
    for split in ['train', 'val']:
        mmcv.mkdir_or_exist(osp.join(args.output_dir, split))
        split_list = os.listdir(osp.join(args.input_dir, split))
        for image_name in split_list:
            src_image_path = osp.join(args.input_dir, split, image_name)
            dst_image_path = osp.join(args.output_dir, split, image_name)
            print(f'generating pseudo label from {src_image_path} to {dst_image_path} ...')
            result = generate_pseudo_label(model, src_image_path, threshold=args.threshold)[0]
            cv2.imwrite(dst_image_path, result)

    print('Done!')


if __name__ == '__main__':
    main()
