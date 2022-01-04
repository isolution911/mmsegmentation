# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, generate_pseudo_label
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser(prog='infer', usage='image(s) inference', description='MMSegmentation inference')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('-i', '--input_dir', help='Directory of input image(s)')
    parser.add_argument('-o', '--output_dir', help='Directory of output image(s)')
    parser.add_argument(
        '--device', default='cuda:9', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='Water',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # image list
    image_list = os.listdir(args.input_dir)
    mmcv.mkdir_or_exist(args.output_dir)
    for image in image_list:
        src_image_path = os.path.join(args.input_dir, image)
        dst_image_path = os.path.join(args.output_dir, image)
        print(f'inferring {src_image_path} to {dst_image_path} ...')
        result = inference_segmentor(model, src_image_path)
        # result = generate_pseudo_label(model, src_image_path, threshold=0.9)
        model.show_result(
            img=src_image_path,
            result=result,
            palette=get_palette(args.palette),
            show=False,
            wait_time=0,
            out_file=dst_image_path,
            opacity=args.opacity
        )

    print('Done!')


if __name__ == '__main__':
    main()
