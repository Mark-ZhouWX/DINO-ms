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
# import torch
# from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(bbox) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.unbind(-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return ops.stack(new_bbox, axis=-1)


def box_xyxy_to_cxcywh(bbox) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    x0, y0, x1, y1 = bbox.unbind(-1)
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
    x1 = boxes[:, 0].clamp(min=0, max=w)
    y1 = boxes[:, 1].clamp(min=0, max=h)
    x2 = boxes[:, 2].clamp(min=0, max=w)
    y2 = boxes[:, 3].clamp(min=0, max=h)
    boxes = ops.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def box_scale(boxes, scale_x: float, scale_y: float) -> Tensor:
    """
    Scale the box with horizontal and vertical scaling factors
    """
    boxes[:, 0::2] *= scale_x
    boxes[:, 1::2] *= scale_y
    return boxes


def box_iou(boxes1, boxes2) -> Tuple[Tensor]:
    """Modified from ``torchvision.ops.box_iou``

    Return both intersection-over-union (Jaccard index) and union between
    two sets of boxes.

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes, in x1,y1,x2,y2 format
        boxes2: (torch.Tensor[M, 4]): second set of boxes, in x1,y1,x2,y2 format

    Returns:
        Tuple: A tuple of NxM matrix, with shape `(torch.Tensor[N, M], torch.Tensor[N, M])`,
        containing the pairwise IoU and union values
        for every element in boxes1 and boxes2.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = ops.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # left right [N,M,2]
    rb = ops.maximum(boxes1[:, None, 2:], boxes2[:, None, 2:])  # right bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2[:, None] - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2) -> Tensor:
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

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = ops.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = ops.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


def masks_to_boxes(masks) -> Tensor:
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is
    the number of masks, (H, W) are the spatial dimensions.

    Args:
        masks (Tensor[N, H, W]):
    Returns:
        torch.Tensor: a [N, 4] tensor with
        the boxes in (x0, y0, x1, y1) format.
    """
    if masks.numel() == 0:
        return ops.zeros((0, 4), masks.dtype)

    h, w = masks.shape[-2:]

    y = ops.arange(0, h, dtype=ms.float32)
    x = ops.arange(0, w, dtype=ms.float32)
    y, x = ops.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return ops.stack([x_min, y_min, x_max, y_max], 1)
