# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import distance2bbox, bbox_overlaps
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from .gfl_head import GFLHead
from ..task_modules.samplers import PseudoSampler
from ..utils import (multi_apply,
                     unpack_gt_instances)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@MODELS.register_module()
class GFLHeadIncrementERD(GFLHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 4.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Default: dict(type='GN', num_groups=32,
            requires_grad=True).
        loss_qfl (:obj:`ConfigDict` or dict): Config of Quality Focal Loss
            (QFL).
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
             to 'DistancePointBBoxCoder'.
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max}``
            in QFL setting. Defaults to 16.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_dfl: ConfigType = dict(
                     type='DistributionFocalLoss', loss_weight=0.25),
                 loss_ld: ConfigType = dict(
                     type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 reg_max: int = 16,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='gfl_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,
            **kwargs)

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_ld = MODELS.build(loss_ld)

    def distill_loss_by_image_single(self,
                                     anchors,
                                     new_cls_scores,
                                     new_bbox_preds,
                                     ori_cls_inds,
                                     ori_box_inds,
                                     ori_cls_scores,
                                     ori_bbox_preds,
                                     dist_loss_weight,
                                     ori_num_classes: int, avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # ===========>  distillation classification (only u+2 * sigma) using l2 loss
        new_topk_cls_scores = new_cls_scores.gather(0,
                                                    ori_cls_inds.unsqueeze(-1).expand(-1, new_cls_scores.size(-1)))
        ori_topk_cls_scores = ori_cls_scores.gather(0,
                                                    ori_cls_inds.unsqueeze(-1).expand(-1, ori_cls_scores.size(-1)))

        loss_dist_cls = dist_loss_weight * self.l2_loss(new_topk_cls_scores, ori_topk_cls_scores)

        # ===========>  distillation regression (only u+2 * sigma) using ld loss
        anchor_centers = self.anchor_center(anchors)
        # ori decode bbox, shape (Num,4)
        ori_bbox_preds_tblr = self.integral(ori_bbox_preds)
        decode_bbox_pred = distance2bbox(anchor_centers, ori_bbox_preds_tblr)

        ori_cls_conf = ori_cls_scores.sigmoid()
        cls_conf, ids = ori_cls_conf.max(dim=-1)

        # nms
        nms_cfg = dict(iou_threshold=0.005)  # 0.005

        thr_bboxes, thr_scores, thr_id = decode_bbox_pred[ori_box_inds], cls_conf[ori_box_inds], \
                                         ids[ori_box_inds]
        _, keep = batched_nms(thr_bboxes, thr_scores, thr_id, nms_cfg)

        nms_bbox_preds = new_bbox_preds.gather(
            0, ori_box_inds.unsqueeze(-1).expand(-1, new_bbox_preds.size(-1)))
        new_topk_bbox_preds = nms_bbox_preds.gather(
            0, keep.unsqueeze(-1).expand(-1, nms_bbox_preds.size(-1)))

        nms_ori_topk_bbox_preds = ori_bbox_preds.gather(
            0, ori_box_inds.unsqueeze(-1).expand(-1, ori_bbox_preds.size(-1)))
        ori_topk_bbox_preds = nms_ori_topk_bbox_preds.gather(
            0, keep.unsqueeze(-1).expand(-1, nms_ori_topk_bbox_preds.size(-1)))

        new_topk_bbox_corners = new_topk_bbox_preds.reshape(-1, self.reg_max + 1)
        ori_topk_pred_corners = ori_topk_bbox_preds.reshape(-1, self.reg_max + 1)

        weight_targets = new_cls_scores.reshape(-1, ori_num_classes)[ori_box_inds].detach().sigmoid()
        weight_targets = weight_targets.max(dim=1)[0][keep.reshape(-1)]
        loss_dist_bbox = dist_loss_weight * self.loss_ld(new_topk_bbox_corners, ori_topk_pred_corners,
                                                         weight=weight_targets[:, None].expand(-1, 4).reshape(
                                                             -1), avg_factor=4.0)

        return loss_dist_cls, loss_dist_bbox

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            stride: Tuple[int], ori_num_classes: int, avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        # cls_score = cls_score.permute(0, 2, 3,
        #                               1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score[:, ori_num_classes:].permute(0, 2, 3,
                                                           1).reshape(-1, self.cls_out_channels - ori_num_classes)

        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes - ori_num_classes  # only optimize the novel classes
        labels[labels == self.num_classes] = bg_class_ind  # only optimize the novel classes

        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    @staticmethod
    def l2_loss(pred, target, reduction='mean'):
        r"""Function that takes the mean element-wise square value difference.
        """
        assert target.size() == pred.size()
        loss = (pred - target).pow(2).float()
        if reduction != 'none':
            loss = torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        return loss

    def loss_by_feat(self,
                     ori_outs: Tuple[Tensor],
                     new_outs: Tuple[Tensor],
                     ori_topk_cls_inds,  # for distillation
                     ori_topk_cls_scores,  # for distillation
                     ori_topk_bbox_inds,  # for distillation
                     ori_topk_bbox_preds,  # for distillation
                     ori_num_classes,
                     dist_loss_weight,
                     model,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # ****************************** ori loss **********************************
        cls_scores, bbox_preds = new_outs
        num_imgs = cls_scores[0].size(0)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl, \
        avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            ori_num_classes=ori_num_classes,
            avg_factor=avg_factor)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        # ****************************** distill loss **********************************
        anchor_list = torch.cat(anchor_list, dim=1)
        bbox_preds_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for bbox_pred in bbox_preds]
        bbox_preds_list = torch.cat(bbox_preds_list, dim=1)

        ori_cls_scores, ori_bbox_preds = ori_outs

        ori_cls_scores_list = [
            ori_cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes)
            for ori_cls_score in ori_cls_scores]
        ori_cls_scores_list = torch.cat(ori_cls_scores_list, dim=1)

        ori_bbox_preds_list = [
            ori_bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for ori_bbox_pred in ori_bbox_preds]
        ori_bbox_preds_list = torch.cat(ori_bbox_preds_list, dim=1)

        new_cls_scores_list = [
            cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes) for cls_score in cls_scores]
        new_cls_scores_list = torch.cat(new_cls_scores_list, dim=1)

        loss_dist_cls, loss_dist_bbox = multi_apply(
            self.distill_loss_by_image_single,
            anchor_list,
            new_cls_scores_list,
            bbox_preds_list,
            ori_topk_cls_inds,
            ori_topk_bbox_inds,
            ori_cls_scores_list,
            ori_bbox_preds_list,
            dist_loss_weight=dist_loss_weight,
            ori_num_classes=ori_num_classes,
            avg_factor=avg_factor)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_dist_cls=loss_dist_cls,
            loss_dist_bbox=loss_dist_bbox)

    # def loss(self, ori_out: Tuple[Tensor], new_out: Tuple[Tensor],batch_data_samples: SampleList) -> dict:
    def loss(self, ori_outs: Tuple[Tensor], new_outs: Tuple[Tensor], batch_data_samples: SampleList,
             topk_cls_inds, topk_cls_scores, topk_bbox_inds, topk_bbox_preds,
             ori_num_classes, dist_loss_weight, model) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = (ori_outs, new_outs, topk_cls_inds, topk_cls_scores, topk_bbox_inds, topk_bbox_preds,
                       ori_num_classes, dist_loss_weight, model) + (
                          batch_gt_instances, batch_img_metas,
                          batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
