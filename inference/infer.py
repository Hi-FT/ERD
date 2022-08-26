# coding=utf-8
"""
@Data: 2020/09/01
@Author: 算影
@Email: wangmang.wm@alibaba-inc.com
"""
import os
import argparse

import numpy as np
import cv2
import onnxruntime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def parse_args():
    parser = argparse.ArgumentParser(description='run detection model')
    parser.add_argument('-m', '--model_file', type=str, required=True, help='model filename including path')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='path of test images')
    parser.add_argument('-s', '--infer_size', type=int, nargs='+', default=[480, 480], help='image size input to model')
    parser.add_argument('-t', '--conf_thr', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('-o', '--out_path', type=str, default='detections', help='confidence threshold')
    parser.add_argument('--onnx', action='store_true', help='infer with ONNX model')
    args = parser.parse_args()
    return args


def pre_processor(image_decoded_list, infer_size):
    """First resize image keeping the aspect ratio, then pad to square.

    Args:
        image_decoded_list (list): list of decoded image array
        infer_size (list): [height, width]

    Returns:
        img_padded_array_list (list): padded image array list
        img_resized_shape_list (list): raw image sized shape list
        scale_list (list): scale factor list
    """
    img_padded_array_list = []
    img_resized_shape_list = []
    scale_list = []
    for img_decoded in image_decoded_list:
        height, width, _ = img_decoded.shape
        scale = max(height / infer_size[0], width / infer_size[1])
        img_raw_resize = int(height / scale), int(width / scale)

        # keeping aspect ratio resize
        new_size = round(width / scale), round(height / scale)
        img_resized = cv2.resize(img_decoded, dsize=new_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img_resized = img_resized - np.array([123.675, 116.28, 103.53], dtype=np.float32)
        img_resized = img_resized / np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # pad to square
        img_padded = np.zeros((*infer_size, 3), dtype=img_resized.dtype)
        img_padded[:new_size[1], :new_size[0], ...] = img_resized
        img_padded = img_padded.transpose((2, 0, 1))  # transpose to (C, H, W)

        img_padded_array_list.append(img_padded)
        img_resized_shape_list.append(img_raw_resize)
        scale_list.append(scale)

    return img_padded_array_list, img_resized_shape_list, scale_list


def ts_model_infer(model_file, img_padded_array_list, device):
    ts_model = torch.jit.load(model_file, map_location=device)
    img_padded_tensor = torch.from_numpy(np.asarray(img_padded_array_list, dtype=np.float32)).to(device)
    outs = ts_model(img_padded_tensor)
    return outs


def onnx_model_infer(model_file, img_padded_array_list):
    ort_model = onnxruntime.InferenceSession(model_file)
    img_padded_np = np.asarray(img_padded_array_list, dtype=np.float32)
    outs = ort_model.run(None, {'INPUT__0': img_padded_np})
    return outs


class AnchorGenerator(object):
    """Standard anchor generator for 2D anchor-based detectors.
    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in propotion to anchors'
            width and height. By default it is 0 in V2.0.
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.
        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.
        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            torch.Tensor: Anchors in a single-level feature maps
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.
        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.
        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.
        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.
        Return:
            list[torch.Tensor]: Anchors in multiple feature levels.
                The sizes of each tensor should be [N, 4], where
                N = width * height * num_base_anchors, width and height
                are the sizes of the corresponding feature lavel,
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level.
        Note:
            This function is usually called by method ``self.grid_anchors``.
        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map.
                Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.
        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
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


def anchor_center(anchors):
    """Get anchor centers from anchors.
    Args:
        anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.
    Returns:
        Tensor: Anchor centers with shape (N, 2), "xy" format.
    """
    anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
    anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
    return torch.stack([anchors_cx, anchors_cy], dim=-1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def nms(dets, thresh):
    if isinstance(dets, torch.Tensor):
        dets_ori = dets
        dets = dets.cpu().numpy()
    elif isinstance(dets, np.ndarray):
        dets = dets

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return torch.from_numpy(dets[keep]).type_as(dets_ori), torch.from_numpy(np.array(keep))


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels

    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]

    dets, keep = nms(torch.cat([bboxes_for_nms, scores[:, None]], 1), 0.6)
    bboxes = bboxes[keep]
    scores = dets[:, -1]
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([bboxes, scores[:, None]], 1), labels


def _get_bboxes_single(
        cls_scores,
        bbox_preds,
        mlvl_anchors,
        img_shape,
        anchor_generator,
        conf_thr,
        scale_factor,
        rescale=False):
    """Transform outputs for a single batch item into labeled boxes.
    Args:
        cls_scores (list[Tensor]): Box scores for a single scale level
            has shape (num_classes, H, W).
        bbox_preds (list[Tensor]): Box distribution logits for a single
            scale level with shape (4*(n+1), H, W), n is max value of
            integral set.
        mlvl_anchors (list[Tensor]): Box reference for a single scale level
            with shape (num_total_anchors, 4).
        img_shape (tuple[int]): Shape of the input image,
            (height, width, 3).
        anchor_generator (object): anchor generator object
        conf_thr (float): confidence threshold
        scale_factor (ndarray): Scale factor of the image arange as
            (w_scale, h_scale, w_scale, h_scale).
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Default: False.
    Returns:
        tuple(Tensor):
            det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                the first 4 columns are bounding box positions
                (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                between 0 and 1.
            det_labels (Tensor): A (N,) tensor where each item is the
                predicted class label of the corresponding box.
    """
    assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
    mlvl_bboxes = []
    mlvl_scores = []
    for cls_score, bbox_pred, stride, anchors in zip(
            cls_scores, bbox_preds, anchor_generator.strides,
            mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        assert stride[0] == stride[1]

        scores = cls_score.permute(1, 2, 0).reshape(
            -1, cls_score.shape[0]).sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0)
        integral = Integral()
        bbox_pred = integral(bbox_pred) * stride[0]

        nms_pre = 1000
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]

        bboxes = distance2bbox(anchor_center(anchors), bbox_pred, max_shape=img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)

    mlvl_bboxes = torch.cat(mlvl_bboxes)
    if rescale:
        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

    mlvl_scores = torch.cat(mlvl_scores)
    # Add a dummy background class to the backend when using sigmoid
    # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    # BG cat_id: num_class
    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

    det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, conf_thr, 100)
    return det_bboxes, det_labels


def get_bboxes(
        cls_scores,
        bbox_preds,
        img_metas,
        anchor_generator,
        conf_thr,
        device,
        cfg=None,
        rescale=False):
    """Transform network output for a batch into bbox predictions.
    Args:
        cls_scores (list[Tensor]): Box scores for each scale level
            Has shape (N, num_anchors * num_classes, H, W)
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W)
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        anchor_generator (object): anchor generator object
        conf_thr (float): confidence threshold
        device (torch.device): tensor device
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used
        rescale (bool): If True, return boxes in original image space.
            Default: False.
    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where the first 4 columns
            are bounding box positions (tl_x, tl_y, br_x, br_y) and the
            5-th column is a score between 0 and 1. The second item is a
            (n,) tensor where each item is the predicted class labelof the
            corresponding box.
    """
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = anchor_generator.grid_anchors(featmap_sizes, device=device)

    result_list = []
    for img_id in range(len(img_metas)):
        cls_score_list = [
            cls_scores[i][img_id].detach() for i in range(num_levels)
        ]
        bbox_pred_list = [
            bbox_preds[i][img_id].detach() for i in range(num_levels)
        ]
        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']
        proposals = _get_bboxes_single(cls_score_list, bbox_pred_list,
                                       mlvl_anchors, img_shape, anchor_generator, conf_thr,
                                       scale_factor, rescale)
        result_list.append(proposals)
    return result_list


def main():
    args = parse_args()
    image_path = args.image_path
    infer_size = args.infer_size
    model_file = args.model_file
    conf_thr = args.conf_thr
    out_path = args.out_path

    # ------- preprocess -------
    image_files = [img for img in os.listdir(image_path) if img.endswith(('.jpg', '.png'))]
    image_decoded_list_all = [cv2.imread(os.path.join(image_path, image_file)) for image_file in image_files]

    # set max batch size to 20
    max_batch_size = 20
    for idx in range(0, len(image_decoded_list_all), max_batch_size):
        start_idx = idx
        end_idx = min(len(image_decoded_list_all), idx + max_batch_size)
        image_decoded_list = image_decoded_list_all[start_idx:end_idx]
    
        img_padded_array_list, img_resized_shape_list, scale_list = pre_processor(image_decoded_list, infer_size)

        # ------- model infernce -------
        if args.onnx:
            device = torch.device('cpu')
            outs_ort = onnx_model_infer(model_file, img_padded_array_list)
            cls_scores = [torch.from_numpy(i) for i in outs_ort[:5]]
            bbox_preds = [torch.from_numpy(i) for i in outs_ort[5:]]
            outs = (cls_scores, bbox_preds)  # cls_scores, bbox_preds
        else:
            if torch.cuda.is_available():
                device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
            else:
                device = torch.device('cpu')
            outs_ts = ts_model_infer(model_file, img_padded_array_list, device)  # tuple[Tensor]
            cls_scores, bbox_preds = outs_ts[:5], outs_ts[5:]
            outs = (cls_scores, bbox_preds)  # cls_scores, bbox_preds

        # ------- postprocess -------
        img_metas = []
        for img_resized_shape, scale_factor in zip(img_resized_shape_list, scale_list):
            img_metas.append({'img_shape': img_resized_shape, 'scale_factor': 1 / scale_factor})

        # generate anchors
        anchor_generator = AnchorGenerator(ratios=[1.0], octave_base_scale=8, scales_per_octave=1,
                                        strides=[8, 16, 32, 64, 128])
        # get bbox list
        bbox_list = get_bboxes(*outs, img_metas, anchor_generator, conf_thr, device, rescale=True)
        
        # save detections to image
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i, image_decoded in enumerate(image_decoded_list):
            pred = bbox_list[i]
            bbox_score_pred_list = pred[0].cpu().tolist()
            label_pred_list = pred[1].cpu().tolist()
            if not bbox_score_pred_list:
                continue
            for j, bbox_score_pred in enumerate(bbox_score_pred_list):
                bbox_pred = bbox_score_pred[:4]
                score_pred = bbox_score_pred[4]
                label_pred = label_pred_list[j]
                # plot bbox
                cv2.rectangle(image_decoded, (round(bbox_pred[0]), round(bbox_pred[1])), (round(bbox_pred[2]), round(bbox_pred[3])), (0, 255, 0), 2)
                # add label and score
                cx = round(bbox_pred[0])
                cy = round(bbox_pred[1]) - 12
                cv2.putText(image_decoded, '{}|{:.3f}'.format(label_pred, score_pred), (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
            cv2.imwrite(os.path.join(out_path, os.path.splitext(image_files[i])[0] + '.jpg'), image_decoded)


if __name__ == '__main__':
    main()
