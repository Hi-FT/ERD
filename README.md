## ERD

#### Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation
<p align='left'>
  <img src='figs/framework.jpg' width='721'/>
</p>

The code is coming soon！

## Get Started

Please see [GETTING_STARTED.md](https://github.com/open-mmlab/mmdetection/blob/v2.6.0/docs/get_started.md) for the basic usage of MMDetection.

## Train
```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'

tools/train.py configs/gfl_incre/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats.py --work-dir=/model_zoo/mmdet/gfl_incre/first_40_incre_last_40/
```

## Test
tools/test.py configs/gfl_incre/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats.py /model_zoo/mmdet/gfl_incre/first_40_incre_last_40/latest.pth 8 --eval bbox --options classwise=True

## Citation
Please cite the following paper if this repo helps your research:
```bibtex
@InProceedings{ERD,
    author    = {Tao Feng and Mang Wang and Hangjie Yuan},
    title     = {Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2022}
}
```
