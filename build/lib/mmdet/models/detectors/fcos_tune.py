# coding=utf-8
"""
@Data: 2021/1/5
@Author: 算影
@Email: wangmang.wm@alibaba-inc.com
"""
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

import os
import torch
import warnings
import mmcv
from collections import OrderedDict
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.parallel import MMDistributedDataParallel
from mmdet.core import distance2bbox


@DETECTORS.register_module()
class FCOSTune(SingleStageDetector):
    """Finetune anchor-free object detector based on FCOS.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ori_checkpoint_file=None,
                 ori_num_classes=40):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)
        self.ori_num_classes = ori_num_classes
        # init original branchs of new model
        self.load_checkpoint_for_new_model(ori_checkpoint_file)

    def load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=False, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                          v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        added_branch_weight = self.bbox_head.conv_cls.weight[self.ori_num_classes:, ...]
        added_branch_bias = self.bbox_head.conv_cls.bias[self.ori_num_classes:, ...]
        state_dict['bbox_head.conv_cls.weight'] = torch.cat(
            (state_dict['bbox_head.conv_cls.weight'], added_branch_weight), dim=0)
        state_dict['bbox_head.conv_cls.bias'] = torch.cat(
            (state_dict['bbox_head.conv_cls.bias'], added_branch_bias), dim=0)
        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)
