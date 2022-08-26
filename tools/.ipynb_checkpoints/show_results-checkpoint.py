import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.models.detectors import BaseDetector


config_file = 'configs/gfl_incre/gfl_r50_fpn_1x_coco_first_40_cats.py'
checkpoint_file = '/data-nas/ss/model_zoo/mmdet/gfl_incre/gfl_r50_fpn_1x_coco_first_40_cats/epoch_12.pth'

model = init_detector(config_file,checkpoint_file)

img_dir = '/data-nas/ss/fea_save/JPEGImage/'
out_dir = '/data-nas/ss/fea_save/show_results/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

img =  '/data-nas/ss/fea_save/JPEGImage/000000289586.jpg'
result = inference_detector(model,img)
show_result_pyplot(model, img, result, score_thr=0.3)

print(result)