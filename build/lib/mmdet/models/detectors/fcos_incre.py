# coding=utf-8
"""
@Data: 2021/1/4
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
class FCOSIncre(SingleStageDetector):
    """Incremental anchor-free object detector based on FCOS.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ori_config_file=None,
                 ori_checkpoint_file=None,
                 ori_num_classes=40,
                 top_k=100,
                 dist_loss_weight=1):
        super().__init__(backbone, neck, bbox_head, train_cfg,
                         test_cfg, pretrained)
        self.ori_checkpoint_file = ori_checkpoint_file
        self.ori_num_classes = ori_num_classes
        self.top_k = top_k
        self.dist_loss_weight = dist_loss_weight
        self.init_detector(ori_config_file, ori_checkpoint_file)

    def _load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=False, logger=None):
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

    def init_detector(self, config, checkpoint_file):
        """Initialize detector from config file.

        Args:
            config (str): Config file path or the config
                object.
            checkpoint_file (str): Checkpoint path. If left as None, the model
                will not load any weights.

        Returns:
            nn.Module: The constructed detector.
        """
        assert os.path.isfile(checkpoint_file), '{} is not a valid file'.format(checkpoint_file)
        ##### init original model & frozen it #####
        # build model
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.model.bbox_head.num_classes = self.ori_num_classes
        ori_model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        # load checkpoint
        load_checkpoint(ori_model, checkpoint_file)
        # set to eval mode
        ori_model.eval()
        ori_model.forward = ori_model.forward_dummy
        # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        ##### init original branchs of new model #####
        self._load_checkpoint_for_new_model(checkpoint_file)

        self.ori_model = ori_model

    def model_forward(self, img):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            img (Tensor): Input to the model.

        Returns:
            outs (Tuple(List[Tensor])): Three model outputs.
                # cls_scores (List[Tensor]): Classification scores for each FPN level.
                # bbox_preds (List[Tensor]): BBox predictions for each FPN level.
                # centernesses (List[Tensor]): Centernesses predictions for each FPN level.
        """
        # forward the model without gradients
        with torch.no_grad():
            outs = self.ori_model(img)
        return outs

    def sel_pos(self, cls_scores, bbox_preds, centernesses):
        """Select positive predictions based on classification scores.

        Args:
            model (nn.Module): The loaded detector.
            cls_scores (List[Tensor]): Classification scores for each FPN level.
            bbox_preds (List[Tensor]): BBox predictions for each FPN level.
            centernesses (List[Tensor]): Centernesses predictions for each FPN level.

        Returns:
            cat_cls_scores (Tensor): FPN concatenated classification scores.
            cat_centernesses (Tensor): FPN concatenated centernesses.
            topk_bbox_preds (Tensor): Selected top-k bbox predictions.
            topk_inds (Tensor): Selected top-k indices.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        # concat cls_scores, bbox_preds and centerness
        num_imgs = cls_scores[0].size(0)
        cat_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.ori_model.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ]
        cat_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        cat_centernesses = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for centerness in centernesses
        ]
        cat_cls_scores = torch.cat(cat_cls_scores, dim=1)
        cat_bbox_preds = torch.cat(cat_bbox_preds, dim=1)
        cat_centernesses = torch.cat(cat_centernesses, dim=1)
        # select indices of top-k confidences
        cat_conf = cat_cls_scores.sigmoid() * cat_centernesses.sigmoid()[..., None]
        max_scores, _ = cat_conf.max(dim=-1)
        _, topk_inds = max_scores.topk(self.top_k, dim=1)
        topk_cls_scores = cat_cls_scores.gather(
            1, topk_inds.unsqueeze(-1).expand(-1, -1, cat_cls_scores.size(-1)))
        topk_bbox_preds = cat_bbox_preds.gather(
            1, topk_inds.unsqueeze(-1).expand(-1, -1, cat_bbox_preds.size(-1)))
        topk_centerness_preds = cat_centernesses.gather(1, topk_inds)

        return topk_cls_scores, topk_bbox_preds, topk_centerness_preds, topk_inds

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels):
        # get original model outputs
        ori_outs = self.model_forward(img)
        # select positive predictions from original model
        topk_cls_scores, topk_bbox_preds, topk_centerness_preds, topk_inds = self.sel_pos(*ori_outs)

        # get new model outputs
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        # calculate losses including general losses of new model and distillation losses of original model
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas) + \
            (topk_cls_scores, topk_bbox_preds, topk_centerness_preds, topk_inds,
             self.ori_num_classes, self.dist_loss_weight, self)
        losses = self.bbox_head.loss(*loss_inputs)
        return losses
