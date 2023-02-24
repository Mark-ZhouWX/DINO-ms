import os

import cv2
import mindspore as ms
import numpy as np
from mindspore import nn, ops, Tensor

from common.dataset.transform import get_size_with_aspect_ratio
from common.utils.box_ops import box_xyxy_to_cxcywh
from common.utils.preprocessing import pad_as_batch
from common.utils.system import is_windows
from model_zoo.dino.build_model import build_dino


def get_input():
    # test inference runtime
    image_root = r"C:\02Data\demo\image" if is_windows else '/data1/zhouwuxing/demo/'
    image_path1 = os.path.join(image_root, 'hrnet_demo.jpg')
    image_path2 = os.path.join(image_root, 'road554.png')
    image_path3 = os.path.join(image_root, 'orange_71.jpg')

    inputs_list = [dict(image=Tensor.from_numpy(cv2.imread(image_path1)).transpose(2, 0, 1),
                        instances=dict(image_size=(423, 359), gt_classes=Tensor([3, 7], ms.int32),
                                       gt_boxes=Tensor([[100, 200, 210, 300], [50, 100, 90, 150]]))),
                   dict(image=Tensor.from_numpy(cv2.imread(image_path2)).transpose(2, 0, 1),
                        instances=dict(image_size=(400, 300), gt_classes=Tensor([21, 45, 9], ms.int32),
                                       gt_boxes=Tensor(
                                           [[80, 220, 150, 320], [180, 100, 300, 200], [150, 150, 180, 180]]))),
                   # dict(image=Tensor.from_numpy(cv2.imread(image_path3)).transpose(2, 0, 1),
                   #      instances=dict(image_size=(1249, 1400), gt_classes=Tensor([3, 7]),
                   #                     gt_boxes=Tensor([[100, 200, 210, 300], [50, 100, 90, 150]]))),
                   ]
    return inputs_list, image_root


class Resize(object):
    def __init__(self, size=800, max_size=960):
        self.size = size
        self.max_size = max_size

    def __call__(self, img: Tensor, boxes: Tensor):
        if self.size is None:
            print(f'no resize')
            return img, boxes
        img = img.asnumpy().transpose(1, 2, 0)
        h, w, _ = img.shape

        nh, nw = get_size_with_aspect_ratio(img.shape, self.size, self.max_size)
        resize_pad_img = cv2.resize(img, (nw, nh), cv2.INTER_CUBIC)

        # modify boxes
        ratio_width, ratio_height = float(nw)/float(w), float(nh)/float(h)
        boxes = boxes * Tensor([ratio_width, ratio_height, ratio_width, ratio_height])

        resize_pad_img = Tensor(resize_pad_img).transpose(2, 0, 1)
        return resize_pad_img, boxes


class Pad(object):
    def __init__(self, tgt_h, tgt_w):
        self.tgt_h = tgt_h
        self.tgt_w = tgt_w

    def __call__(self, img):
        c, h, w = img.shape
        if self.tgt_h is None or self.tgt_w is None:
            print(f'no pad')
            new_img = img
            new_mask = ops.zeros((h, w), ms.float32)
            return new_img, new_mask
        new_img = ops.zeros((c, self.tgt_h, self.tgt_w), ms.float32)
        new_mask = ops.ones((self.tgt_h, self.tgt_w), ms.float32)
        new_img[:, :h, :w] = img
        new_mask[:h, :w] = 0
        return new_img, new_mask


def convert_input_format_with_resizepad(batched_inputs):
    batched_inputs = [batched_inputs[0]]
    images = [x['image'] for x in batched_inputs]
    pixel_mean = Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
    pixel_std = Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    org_sizes = Tensor([[img.shape[1], img.shape[2]] for img in images])

    # targets
    resize = Resize(800, 960)
    pad = Pad(960, 960)
    print(f'pad size', pad.tgt_w, pad.tgt_h)
    gt_instances = [x["instances"] for x in batched_inputs]
    new_targets = []
    gt_classes_list = []
    gt_boxes_list = []
    gt_valids_list = []
    unpad_img_sizes_list = []
    new_image_list = []
    mask_list = []
    for image, targets_per_image in zip(images, gt_instances):
        # h, w = targets_per_image['image_size']
        image, targets_per_image['gt_boxes'] = resize(image, targets_per_image['gt_boxes'])
        _, h, w = image.shape
        print('resized', h, w)
        # Norm for box
        image_size_xyxy = Tensor([w, h, w, h], dtype=ms.float32)
        gt_classes = targets_per_image['gt_classes']
        gt_boxes = targets_per_image['gt_boxes'] / image_size_xyxy  # with reference to valid w,h

        gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
        new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        # print('before size', image_size_xyxy)
        # print('before box', targets_per_image['gt_boxes'])

        # Norm and pad for image
        image = normalizer(image)
        image, mask = pad(image)
        new_image_list.append(image)
        mask_list.append(mask)
        num_inst = len(gt_boxes)
        gt_classes_list.append(gt_classes)
        gt_boxes_list.append(gt_boxes)
        gt_valids_list.append(ops.ones(num_inst, type=ms.bool_))
        unpad_img_sizes_list.append([h, w])

    images = ops.stack(new_image_list, 0)
    img_masks = ops.stack(mask_list, 0)

    return images, img_masks, gt_classes_list, gt_boxes_list, gt_valids_list, org_sizes


def convert_input_format(batched_inputs):
    batch_size = len(batched_inputs)

    # images
    pixel_mean = Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
    pixel_std = Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]
    images, unpad_img_sizes = pad_as_batch(images)

    _, _, h, w = images.shape
    img_masks = ops.ones((batch_size, h, w), type=images.dtype)
    for img_id in range(batch_size):
        img_h, img_w = batched_inputs[img_id]["instances"]['image_size']
        img_masks[img_id, :img_h, : img_w] = 0

    # targets
    gt_instances = [x["instances"] for x in batched_inputs]
    new_targets = []
    gt_classes_list = []
    gt_boxes_list = []
    gt_valids_list = []
    for targets_per_image in gt_instances:
        h, w = targets_per_image['image_size']
        image_size_xyxy = Tensor([w, h, w, h], dtype=ms.float32)
        gt_classes = targets_per_image['gt_classes']
        gt_boxes = targets_per_image['gt_boxes'] / image_size_xyxy  # with reference to valid w,h
        gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
        new_targets.append({"labels": gt_classes, "boxes": gt_boxes})

        num_inst = len(gt_boxes)
        gt_classes_list.append(gt_classes)
        gt_boxes_list.append(gt_boxes)
        gt_valids_list.append(ops.ones(num_inst, type=ms.bool_))

    return images, img_masks, gt_classes_list, gt_boxes_list, gt_valids_list


if __name__ == "__main__":
    # set context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU' if is_windows else 'GPU')

    train = True
    infer = False

    pth_dir = r"C:\02Data\models" if is_windows else '/data1/zhouwuxing/pretrained_model/'
    pth_path = os.path.join(pth_dir, "dino_r50_4scale_12ep_49_2AP.pth")
    ms_pth_path = os.path.join(pth_dir, "ms_dino_r50_4scale_12ep_49_2AP.ckpt")

    dino = build_dino(unit_test=True)
    ms.load_checkpoint(ms_pth_path, dino)

    inputs, _ = get_input()
    images, img_masks, gt_classes_list, gt_boxes_list, gt_valids_list = convert_input_format(inputs)
    if infer:
        dino.set_train(False)
        inf_result = dino(inputs)
        print('batch size', len(inf_result))
        for r in inf_result:
            r = r['instances']
            print("image size", r['image_size'])
            print("box shape", r['pred_boxes'].shape)
            print("score shape", r['scores'].shape)
            print("class shape", r['pred_classes'].shape)

    if train:
        # train
        dino.set_train(True)
        # loss_dict = dino(inputs)
        loss_dict = dino(images, img_masks, gt_boxes_list, gt_classes_list, gt_valids_list)
        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                print(key, value)
        else:
            print('loss', loss_dict)

    # train one step
    pass
