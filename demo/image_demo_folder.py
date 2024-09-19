from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import os
import cv2

def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image floder')
    parser.add_argument('out_dir', help='Out floder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # img_dir = '/data-nas/ss/fea_save/val2017/'
    # out_dir = '/data-nas/ss/fea_save/ERD_results/'
 
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    test_list = os.listdir(args.img_dir)
 
    count = 0
    imgs = []
    for test in test_list:
        name = args.img_dir + test
        
        count += 1
        print('model is processing the {}/{} images.'.format(count, len(test_list)))
        result = inference_detector(model, name)
        show_result_pyplot(model, name, result, score_thr=args.score_thr)
        # cv2.imwrite("{}/{}.jpg".format(args.out_dir, test), img)

if __name__ == '__main__':
    main()
