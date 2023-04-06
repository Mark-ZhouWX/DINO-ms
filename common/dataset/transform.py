# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import random
import cv2
import numpy as np


def box_xyxy_to_cxcywh(x):
    """box xyxy to cxcywh"""
    x0, y0, x1, y1 = np.array_split(x.T, 4)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)[0]


def box_cxcywh_to_xyxy(x):
    """box cxcywh to xyxy"""
    x_c, y_c, w, h = np.array_split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1).squeeze(-2)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = np.flip(img, 1)
            _, w, _ = img.shape

            target = target.copy()
            boxes = target["boxes"]
            boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([w, 0, w, 0])
            target["boxes"] = boxes
        return img, target


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    """get size with aspect ratio"""
    h, w, _ = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(max_size * min_original_size / max_original_size)

    if (w <= h and w == size) or (h <= w and h == size):
        return h, w

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return oh, ow


class Resize(object):
    def __init__(self, max_size, size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, img, target):
        h, w, _ = img.shape

        nh, nw = get_size_with_aspect_ratio(img.shape, self.size, self.max_size)
        resize_pad_img = cv2.resize(img, (nw, nh), cv2.INTER_CUBIC)

        target = target.copy()
        # modify boxes
        ratio_width, ratio_height = float(nw)/float(w), float(nh)/float(h)
        boxes = target['boxes']
        boxes = boxes * np.array([ratio_width, ratio_height, ratio_width, ratio_height])
        target['boxes'] = boxes

        # modify size
        target['size'] = (nh, nw)

        return resize_pad_img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        return Resize(max_size=self.max_size, size=size)(img, target)


class Pad(object):
    def __init__(self, tgt_h, tgt_w):
        self.tgt_h = tgt_h
        self.tgt_w = tgt_w

    def __call__(self, img, target):
        h, w, c = img.shape
        new_img = np.zeros((self.tgt_h, self.tgt_w, c), dtype=np.float32)
        new_img[:h, :w, :] = img
        new_mask = np.ones((self.tgt_h, self.tgt_w), dtype=np.float32)
        new_mask[:h, :w] = 0
        target['mask'] = new_mask
        target['size'] = (self.tgt_h, self.tgt_w)
        return new_img, target


class RandomSizeCrop(object):
    """random size crop"""
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, ori_target):
        img_h, img_w, _ = img.shape
        w = random.randint(self.min_size, min(img_w, self.max_size))
        h = random.randint(self.min_size, min(img_h, self.max_size))
        i = np.random.randint(0, img_h - h + 1)
        j = np.random.randint(0, img_w - w + 1)

        cropped_image = img[i: i + h, j: j + w]

        target = ori_target.copy()
        target["size"] = np.array([h, w])
        bboxes = target['boxes']
        max_size = np.array([w, h])
        cropped_boxes = bboxes - np.array([j, i, j, i])
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clip(0)
        target['boxes'] = cropped_boxes.reshape(-1, 4)
        keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)

        target['boxes'] = target['boxes'][keep]
        target['labels'] = target['labels'][keep]
        if len(target['labels']) == 0:
            return img, ori_target
        return cropped_image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, target):
        image = (image / 255)
        image = (image - self.mean) / self.std
        h, w, _ = image.shape

        target = target.copy()
        boxes = target["boxes"]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
        target["boxes"] = boxes
        return image, target


class OutData(object):
    """
    pad image and gt values(label and bbox)
    Returns:
        padded image, padded gt and their masks
    """
    def __init__(self, is_training=True, max_size=1333, pad_label=-1, num_dn=10):
        self.is_training = is_training
        self.pad_max_number = 100
        self.pad_label = pad_label
        self.pad_func = Pad(max_size, max_size)
        self.num_dn = num_dn

    def __call__(self, img, target):

        # pad image
        img, target = self.pad_func(img, target)
        img_data = img.transpose((2, 0, 1)).astype(np.float32)
        mask = target['mask']  # (max_size, max_size) 0 keep, 1 drop
        #
        if self.is_training:
            boxes = target['boxes'].astype(np.float32)
            labels = target['labels'].astype(np.int32)

            box_num = len(labels)
            gt_box = np.pad(boxes, ((0, self.pad_max_number - box_num), (0, 0)), mode="constant", constant_values=0)
            # default_boxes = np.array([[0.5, 0.5, 0.1, 0.1]]).repeat(self.pad_max_number - box_num, axis=0).astype(np.float32)
            # gt_box = np.concatenate([boxes, default_boxes])
            gt_label = np.pad(labels, (0, self.pad_max_number - box_num),
                              mode="constant", constant_values=self.pad_label)
            gt_valid = np.zeros((self.pad_max_number,))
            gt_valid[:box_num] = 1
            gt_valid = gt_valid.astype(np.bool_)  # (pad_max_number) False keep, True drop

            # # fix boxes num = 3
            # gt_box = np.array([0.4, 0.5, 0.1, 0.2], [0.7, 0.6, 0.3, 0.2], [0.3, 0.6, 0.1, 0.2], dtype=np.float32)
            # gt_label = np.array([4, 8, 10], dtype=np.int32)
            # gt_valid = np.ones(3, dtype=np.bool_)
            # print('gt_label', gt_label)
            dn_valid = np.zeros((self.num_dn,), dtype=np.bool_)
            end_index = self.num_dn-(self.num_dn%box_num) if box_num<self.num_dn else self.num_dn
            dn_valid[:end_index] = True
            return img_data, mask, gt_box, gt_label, gt_valid, dn_valid
        else:
            image_id = target['image_id'].astype(np.int32)
            ori_size = np.array(target['ori_size'], dtype=np.int32)
            return img_data, mask, image_id, ori_size
