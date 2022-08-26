# coding=utf-8
"""
@Data: 2021/1/4
@Author: 算影
@Email: wangmang.wm@alibaba-inc.com
"""
import torch
import torch.nn.functional as F

from ..builder import HEADS
from .fcos_head import FCOSHead
from mmcv.runner import force_fp32
from mmdet.core import distance2bbox, reduce_mean

INF = 1e8


@HEADS.register_module()
class FCOSHeadIncre(FCOSHead):
    """Incremental FCOS head for incremental object detection.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):

        # reuse father's constructor
        super().__init__(num_classes, in_channels, regress_ranges, center_sampling, center_sample_radius,
                         norm_on_bbox, centerness_on_reg, loss_cls, loss_bbox, loss_centerness, norm_cfg, **kwargs)

    @staticmethod
    def py_sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):
        r"""Fucntion that sigmoid focal loss
        """
        pred_sigmoid = pred.sigmoid()
        target = F.softmax(target, dim=-1).type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        if reduction != 'none':
            loss = torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        return loss

    @staticmethod
    def l2_loss(pred, target, reduction='mean'):
        r"""Function that takes the mean element-wise square value difference.
        """
        assert target.size() == pred.size()
        loss = (pred - target).pow(2).float()
        if reduction != 'none':
            loss = torch.mean(loss) if reduction == 'mean' else torch.sum(loss)
        return loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses',
                          'ori_topk_cls_scores', 'ori_topk_bbox_preds', 'ori_topk_centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             ori_topk_cls_scores,
             ori_topk_bbox_preds,
             ori_topk_centernesses,
             ori_topk_inds,
             ori_num_classes,
             dist_loss_weight,
             new_model,
             gt_bboxes_ignore=None):
        ####### calculate general losses of added branches of new model #######
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score[:, ori_num_classes:, :, :].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels - ori_num_classes)
            for cls_score in cls_scores
        ]  # only count added branches of new model
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes - ori_num_classes  # minus ori_num_classes
        flatten_labels[flatten_labels == self.num_classes] = bg_class_ind
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        num_pos_denorm = torch.tensor(num_pos, dtype=torch.float, device=bbox_preds[0].device)
        num_pos_denorm = max(reduce_mean(num_pos_denorm).item(), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos_denorm)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets,
                                                   avg_factor=num_pos_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        ####### calculate distillation losses of old model #######
        # # distillation classification using l2 loss
        # new_cls_scores = [
        #     cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(num_imgs, -1, ori_num_classes)  # branches for original model
        #     for cls_score in cls_scores
        # ]
        # new_cls_scores = torch.cat(new_cls_scores, dim=1)
        # loss_dist_cls = dist_loss_weight * self.l2_loss(new_cls_scores, ori_cls_scores)
        # loss_dist_cls = 10**4 * dist_loss_weight * self.py_sigmoid_focal_loss(new_cls_scores, ori_cls_scores)

        # distillation classification (only positive anchor-points) using l2 loss
        new_cls_scores = [
            cls_score[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes)  # branches for original model
            for cls_score in cls_scores
        ]
        new_cls_scores = torch.cat(new_cls_scores, dim=1)
        new_topk_cls_scores = new_cls_scores.gather(
            1, ori_topk_inds.unsqueeze(-1).expand(-1, -1, new_cls_scores.size(-1)))
        loss_dist_cls = dist_loss_weight * \
            self.l2_loss(new_topk_cls_scores, ori_topk_cls_scores)

        # # distillation bbox_pred (only positive anchor-points) using GIoU loss
        # new_bbox_preds = [
        #     bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        #     for bbox_pred in bbox_preds
        # ]
        # new_bbox_preds = torch.cat(new_bbox_preds, dim=1)
        # # anchor points of new model
        # new_points = torch.cat([points.unsqueeze(0).expand(
        #     num_imgs, -1, -1) for points in all_level_points], dim=1)
        # # get top-k bbox predictions of new model
        # topk_points = new_points.gather(
        #     1, ori_topk_inds.unsqueeze(-1).expand(-1, -1, new_points.size(-1)))
        # new_topk_bbox_preds = new_bbox_preds.gather(
        #     1, ori_topk_inds.unsqueeze(-1).expand(-1, -1, new_bbox_preds.size(-1)))
        # # decode top-k bbox predictions of new model
        # decoded_new_topk_bbox_preds = distance2bbox(topk_points.reshape(-1, topk_points.size(-1)),
        #                                             new_topk_bbox_preds.reshape(-1, new_topk_bbox_preds.size(-1)))

        # # decode top-k bbox predictions of original model
        # decoded_ori_topk_bbox_preds = distance2bbox(topk_points.reshape(-1, topk_points.size(-1)),
        #                                             ori_topk_bbox_preds.reshape(-1, ori_topk_bbox_preds.size(-1)))
        # # giou loss
        # loss_dist_bbox = dist_loss_weight * \
        #     self.loss_bbox(decoded_new_topk_bbox_preds,
        #                    decoded_ori_topk_bbox_preds)

        # # distillation centerness (only positive anchor-points) using l2 loss
        # new_centernesses = [
        #     centerness.permute(0, 2, 3, 1).reshape(
        #         num_imgs, -1)  # branches for original model
        #     for centerness in centernesses
        # ]
        # new_centernesses = torch.cat(new_centernesses, dim=1)
        # new_topk_centernesses = new_centernesses.gather(1, ori_topk_inds)
        # loss_dist_centerness = dist_loss_weight * \
        #     self.l2_loss(new_topk_centernesses, ori_topk_centernesses)

        # # distillation feature maps using L2 loss
        # loss_dist_feat = 0
        # for new_param_tuple, ori_param_tuple in zip(new_model.named_parameters(), new_model.ori_model.named_parameters()):
        #     if 'bbox_head.fcos_cls' not in new_param_tuple[0] \
        #         and 'bbox_head.fcos_reg' not in new_param_tuple[0] \
        #         and 'bbox_head.fcos_centerness' not in new_param_tuple[0]:
        #         if new_param_tuple[1].requires_grad is True:
        #             loss_dist_feat += self.l2_loss(new_param_tuple[1], ori_param_tuple[1])
        #     else:
        #         continue
        # loss_dist_feat = dist_loss_weight * loss_dist_feat

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_dist_cls=loss_dist_cls)
