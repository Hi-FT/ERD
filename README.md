## Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation

Official Pytorch implementation for "[Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation](https://arxiv.org/abs/2204.02136)", CVPR 2022.

###  Introduction
Traditional object detectors are ill-equipped for incremental learning. However, fine-tuning directly on a well-trained detection model with only new data will lead to catastrophic forgetting. Knowledge distillation is a flexible way to mitigate catastrophic forgetting. In Incremental Object Detection (IOD), previous work mainly focuses on distilling for the combination of features and responses. However, they under-explore the information that contains in responses. In this paper, we propose a response-based incremental distillation method, dubbed Elastic Response Distillation (ERD), which focuses on elastically learning responses from the classification head and the regression head. Firstly, our method transfers category knowledge while equipping student detector with the ability to retain localization information during incremental learning. In addition, we further evaluate the quality of all locations and provide valuable responses by the Elastic Response Selection (ERS) strategy. Finally, we elucidate that the knowledge from different responses should be assigned with different importance during incremental distillation. Extensive experiments conducted on MS COCO demonstrate our method achieves state-of-the-art result, which substantially narrows the performance gap towards full training. 

<p align='left'>
  <img src='figs/framework.jpg' width='721'/>
</p>

### Requirements
- Python 3.5+
- PyTorch 1.6
- CUDA 10.2
- [mmcv](https://github.com/open-mmlab/mmcv)

### Get Started

This repo is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please refer to [GETTING_STARTED.md](https://github.com/open-mmlab/mmdetection/blob/v2.6.0/docs/get_started.md) for the basic configuration and usage of MMDetection.

### Train
```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in '/dataset/coco/'

python tools/train.py configs/gfl_incre/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats.py --work-dir=/model_zoo/mmdet/gfl_incre/first_40_incre_last_40/
```

### Test
```python
python tools/test.py configs/gfl_incre/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats.py /model_zoo/mmdet/gfl_incre/first_40_incre_last_40/latest.pth 8 --eval bbox --options classwise=True
```

### Citation
Please cite the following paper if this repo helps your research:
```bibtex
@InProceedings{ERD,
    author    = {Tao Feng and Mang Wang and Hangjie Yuan},
    title     = {Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2022}
}
```
