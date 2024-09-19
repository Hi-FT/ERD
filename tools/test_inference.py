import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector


config_file = 'configs/gfl_incre/gfl_r50_fpn_1x_coco_first_40_cats.py'
checkpoint_file = '/model_zoo/mmdet/gfl_incre/gfl_r50_fpn_1x_coco_first_40_cats/latest.pth'

model = init_detector(config_file,checkpoint_file)

img_dir = '/dataset/coco//val2017/'
out_dir = 'work_dirs/'

img = 'work_dirs/000000581781.jpg'
result = inference_detector(model,img)

print('finished')
# if not os.path.exists(out_dir):
#     os.mkdir(out_dir)

# # img = 'test.jpg'
# # result = inference_detector(model,img)
# # show_result(img, result, model.CLASSES, out_file='testOut.jpg')
# #
# # print(result)

# fp = open('work_dirs/test1.txt','r')
# test_list = fp.readlines()

# imgs=[]
# for test_1 in test_list:
#     test_1 = test_1.replace('\n','')
#     name = img_dir + test_1 + '.jpg'
#     imgs.append(name)

# results = []
# # for i,result in enumerate(inference_detector(model,imgs)):
# #     print('model is processing the {}/{} images.'.format(i+1,len(imgs)))
# #     results.append(result)

# count = 0
# for img in imgs:
#     count += 1
#     print('model is processing the {}/{} images.'.format(count,len(imgs)))
#     result = inference_detector(model,img)
#     results.append(result)

# print('\nwriting results to {}'.format('faster_voc.pkl'))
# mmcv.dump(results,out_dir+'faster_voc.pkl')
