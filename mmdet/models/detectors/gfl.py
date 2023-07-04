# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from torch import Tensor
import torch
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class GFL(SingleStageDetector):
    """Implementation of `GFL <https://arxiv.org/abs/2006.04388>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of GFL. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of GFL. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    # for replay method by minimum cost
    def compute_cost_for_memory(self, batch_inputs: Tensor,
                                batch_data_samples: SampleList, cur_class_num) -> Union[dict, list]:
        """Calculate cost for a batch images

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        batch_cost = self.bbox_head.compute_cost_for_memory(x, batch_data_samples, cur_class_num)
        return batch_cost

    def tensor2numpy(self, x):
        return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

    # for replay method by icaRL
    def compute_cost_for_memory_icarl(self, batch_inputs: Tensor,
                                      batch_data_samples: SampleList, cur_class_num) -> Union[dict, list]:
        """Calculate cost for a batch images

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        # batch_cost = self.bbox_head.compute_cost_for_memory_icarl(x, batch_data_samples, cur_class_num)
        batch_cost = torch.cat([per_x.reshape(per_x.shape[0], per_x.shape[1], -1) for per_x in x], dim=2).mean(-1)
        return self.tensor2numpy(batch_cost)

    # importance metric
    def compute_importance_for_replay_v3(self, batch_inputs: Tensor,
                                         batch_data_samples: SampleList, cur_class_num) -> Union[dict, list]:
        """Calculate cost for a batch images

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        batch_importance = self.bbox_head.compute_importance_for_replay_v3(x, batch_data_samples, cur_class_num)
        return batch_importance

    def compute_cost_and_feats_for_replay_v4(self, batch_inputs: Tensor,
                                             batch_data_samples: SampleList, cur_class_num) -> Union[dict, list]:
        """Calculate cost and feats for a batch images

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        batch_feats = torch.cat([per_x.reshape(per_x.shape[0], per_x.shape[1], -1) for per_x in x], dim=2).mean(-1)
        batch_importances = self.bbox_head.compute_importance_for_replay_v4(x, batch_data_samples, cur_class_num)
        return self.tensor2numpy(batch_feats), batch_importances
