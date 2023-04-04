# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------------------
# Utilities for bounding box manipulation and GIoU
# Modified from:
# https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
# ------------------------------------------------------------------------------------------------

from typing import Tuple

import mindspore as ms
from mindspore import Tensor, ops
import mindspore.numpy as ms_np
# import torch
# from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(bbox) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (ms.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = ops.unstack(bbox, axis=-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    aa = ops.stack(new_bbox, axis=-1)
    # factor = Tensor([[   1,    0,    1,    0],
    #                  [   0,    1,    0,    1],
    #                  [-0.5,    0,  0.5,    0],
    #                  [   0, -0.5,    0,  0.5]], bbox.dtype)
    # aa = ops.matmul(bbox, factor)
    return aa
    # return ops.stack(new_bbox, axis=-1)


def box_xyxy_to_cxcywh(bbox) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    x0, y0, x1, y1 = ops.unstack(bbox, axis=-1)
    new_bbox = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return ops.stack(new_bbox, axis=-1)


def box_area(boxes):
    """Computes the area of a set of bounding boxes,

    Args:
        boxes (Tensor[N, 4]): boxes are specified by their (x1, y1, x2, y2) coordinates

    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_clip(boxes, clip_size: Tuple[int, int]) -> Tensor:
    """
    Clip (in place) the boxes by limiting x coordinates to the range [0, width]
    and y coordinates to the range [0, height].

    Args:
        boxes (Tensor[N, 4]): boxes are specified by their (x1, y1, x2, y2) coordinates
        clip_size (height, width): The clipping box's size.
    """
    h, w = clip_size
    x1 = boxes[:, 0].clip(0, w)
    y1 = boxes[:, 1].clip(0, h)
    x2 = boxes[:, 2].clip(0, w)
    y2 = boxes[:, 3].clip(0, h)
    boxes = ops.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def box_scale(boxes, scale) -> Tensor:
    """
    Scale the box with horizontal and vertical scaling factors

    Args:
        boxes (Tensor[N, 4] or [bs, N, 4]): boxes are specified by their (x1, y1, x2, y2) coordinates
        scale (Tensor[2] or [bs, 2]): scale factors for x and y coordinates
    """
    assert len(boxes.shape) in [2, 3]
    if len(boxes.shape) == 2:
        assert len(scale.shape) == 1
    else:
        assert len(scale.shape) == 2
    scale_x, scale_y = scale.unbind(-1)
    new_scale = ops.stack([scale_x, scale_y, scale_x, scale_y], -1)  # (4,) or (bs, 4)
    new_scale = new_scale.unsqueeze(-2)
    boxes *= new_scale
    return boxes


def box_intersection(boxes1, boxes2) -> Tensor:
    """Modified from ``torchvision.ops.box_iou``

    Return both intersection (Jaccard index).

    Args:
        boxes1: (Tensor[N, 4]): first set of boxes, in x1,y1,x2,y2 format (x2>=x1, y2>y1)
        boxes2: (Tensor[M, 4]): second set of boxes, in x1,y1,x2,y2 format

    Returns:
        Tuple: A tuple of NxM matrix, with shape `(torch.Tensor[N, M], torch.Tensor[N, M])`,
        containing the pairwise IoU and union values
        for every element in boxes1 and boxes2.
    """
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    num_box1 = len(boxes1)
    num_box2 = len(boxes2)

    # Caution:, be careful about the maximum operator. if the input is need broadcast, you'd better do it manually.
    # otherwise loss overflow or bankrupt may occur
    lb = ops.maximum(ms_np.tile(boxes1[:, None, :2], (1, num_box2, 1)),
                     ms_np.tile(boxes2[None, :, :2], (num_box1, 1, 1)))  # left bottom [N,M,2]
    rt = ops.minimum(ms_np.tile(boxes1[:, None, 2:], (1, num_box2, 1)),
                     ms_np.tile(boxes2[None, :, 2:], (num_box1, 1, 1)))  # right top [N,M,2]

    # this is the version that would cause loss overflow problem
    # lb = ops.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # left bottom [N,M,2]
    # rt = ops.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # right top [N,M,2]

    wh = ops.clip_by_value(rt - lb, clip_value_min=Tensor(0.0),
                           clip_value_max=Tensor(100.0))  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def box_iou(boxes1, boxes2, eps=1e-6) -> Tuple:
    """Modified from ``torchvision.ops.box_iou``

    Return both intersection-over-union (Jaccard index) and union between
    two sets of boxes.

    Args:
        boxes1: (Tensor[N, 4]): first set of boxes, in x1,y1,x2,y2 format
        boxes2: (Tensor[M, 4]): second set of boxes, in x1,y1,x2,y2 format

    Returns:
        Tuple: A tuple of NxM matrix, with shape `(torch.Tensor[N, M], torch.Tensor[N, M])`,
        containing the pairwise IoU and union values
        for every element in boxes1 and boxes2.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    inter = box_intersection(boxes1, boxes2)

    num_box1 = len(area1)
    num_box2 = len(area2)
    bc_area1 = ms_np.tile(area1[:, None], (1, num_box2)) # (N, M)
    bs_area2 = ms_np.tile(area2[None, :], (num_box1, 1)) # (N, M)

    union = bc_area1 + bs_area2 - inter

    iou = inter / (union + eps)
    return iou, union


def box_mer_area(boxes1, boxes2) -> Tensor:
    """

    Return both maximum-exterior-rectangle area (Jaccard index) between two sets of boxes.

    Args:
        boxes1: (Tensor[N, 4]): first set of boxes, in x1,y1,x2,y2 format
        boxes2: (Tensor[M, 4]): second set of boxes, in x1,y1,x2,y2 format

    Returns:
        Tensor: A tuple of NxM matrix, with shape [N, M], containing the pairwise MER area values
        for every element in boxes1 and boxes2.
    """
    num_box1 = len(boxes1)
    num_box2 = len(boxes2)

    lt = ops.minimum(ms_np.tile(boxes1[:, None, :2], (1, num_box2, 1)),
                     ms_np.tile(boxes2[None, :, :2], (num_box1, 1, 1)))
    rb = ops.maximum(ms_np.tile(boxes1[:, None, 2:], (1, num_box2, 1)),
                     ms_np.tile(boxes2[None, :, 2:], (num_box1, 1, 1)))
    wh = ops.clip_by_value((rb - lt), clip_value_min=Tensor(0.0))  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    return area


def generalized_box_iou(boxes1, boxes2, eps=1e-6) -> Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The input boxes should be in (x0, y0, x1, y1) format

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes
        boxes2: (torch.Tensor[M, 4]): second set of boxes

    Returns:
        Tensor: a NxM pairwise matrix containing the pairwise Generalized IoU
        for every element in boxes1 and boxes2.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2, eps)

    # area of box minimum exterior rectangle (MER)
    area = box_mer_area(boxes1, boxes2)
    corner = area - union
    return iou - corner / (area + eps)
