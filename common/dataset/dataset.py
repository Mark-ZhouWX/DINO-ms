""" DETR dataset"""
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

from __future__ import division

import os
import numpy as np

import cv2
import mindspore.dataset as de
import mindspore.dataset.vision as C
from mindspore.mindrecord import FileWriter

from common.dataset import transform

coco_classes = ['person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife',
                'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

coco_id_dict = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
                5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
                9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
                52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
                58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
                63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
                74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
                87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

assert len(coco_classes) == 80
assert len(coco_id_dict) == 80

coco_cls_dict = {v: k for k, v in coco_id_dict.items()}

coco_catid_to_clsid = {cat_id: i for i, cat_id in enumerate(coco_id_dict.keys())}
coco_clsid_to_catid = {i: cat_id for i, cat_id in enumerate(coco_id_dict.keys())}

def create_coco_label(args, is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = args.coco_path
    data_type = args.val_data_type
    if is_training:
        data_type = args.train_data_type

    # Classes need to train or scripts.
    train_cls = coco_classes
    train_cls_dict = coco_cls_dict

    anno_json = os.path.join(coco_root, "annotations/instances_{}.json".format(data_type))

    coco = COCO(anno_json)
    classes_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classes_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_valid_ids = []
    image_anno_dict = {}
    image_files_dict = {}

    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        for label in anno:
            bbox = label["bbox"]
            class_name = classes_dict[label["category_id"]]
            if class_name in train_cls:
                x1, x2 = bbox[0], bbox[0] + bbox[2]
                y1, y2 = bbox[1], bbox[1] + bbox[3]
                annos.append([x1, y1, x2, y2] + [train_cls_dict[class_name]] + [int(label["iscrowd"])])

        if is_training:
            if annos:
                image_valid_ids.append(img_id)
                image_files_dict[img_id] = image_path
                image_anno_dict[image_path] = np.array(annos)
            else:
                print(f'{img_id} no annotations')
        else:
            image_valid_ids.append(img_id)
            image_files_dict[img_id] = image_path
            if annos:
                image_anno_dict[image_path] = np.array(annos)
            else:
                image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])

    return image_valid_ids, image_files_dict, image_anno_dict


def data_to_mindrecord_byte_image(args, prefix="DETR.mindrecord", is_training=True, file_num=8):
    """Create MindRecord file."""
    mindrecord_dir = args.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)

    image_ids, image_files_dict, image_anno_dict = create_coco_label(args, is_training)

    detr_json = {
        "image_id": {"type": "int32"},
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
    }
    writer.add_schema(detr_json, "detr_json")

    for image_id in image_ids:
        image_name = image_files_dict[image_id]
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        row = {"image_id": image_id, "image": img, "annotation": annos}
        writer.write_raw_data([row])

    writer.commit()


def create_mindrecord(args, rank=0, prefix="DETR.mindrecord", is_training=True):
    print("Start create DETR dataset")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is DETR.mindrecord0, 1, ... file_num.
    mindrecord_dir = args.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args.dataset_file == "coco":
            if os.path.isdir(args.coco_path):
                if not os.path.exists(args.coco_path):
                    print("Please make sure config:coco_root is valid.")
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(args, prefix, is_training)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
    print("CHECKING MINDRECORD FILES DONE!")
    return mindrecord_file


def preprocess_fn(args, image_id, image, image_anno_dict, is_training):
    """Preprocess function for dataset."""
    if is_training:
        max_h_arr = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        trans = transform.Compose([
            transform.RandomHorizontalFlip(),
            transform.RandomSelect(
                transform.RandomResize(max_h_arr, args.max_size),
                transform.Compose([
                    transform.RandomResize([400, 500, 600]),
                    transform.RandomSizeCrop(384, 600),
                    transform.RandomResize(max_h_arr, max_size=args.max_size),
                ])
            ),
            # normalize both image and boxes
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # pad image and get mask
        out_data = transform.OutData(is_training=True, max_size=args.max_size, num_dn=args.num_dn)
    else:
        trans = transform.Compose([
            # resize image, boxes value updated
            transform.Resize(size=800, max_size=args.max_size),
            # normalize both image and boxes(0,1 value with respect to the valid HW)
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # pad image and gt, then get masks of both
        print(f'num dn in dataset is {args.num_dn}')
        out_data = transform.OutData(is_training=False, max_size=args.max_size)

    image_shape = image.shape[:2]
    ori_shape = image_shape
    gt_box = image_anno_dict[:, :4]
    # gt_label = image_anno_dict[:, 4]
    gt_label = np.vectorize(coco_catid_to_clsid.get)(image_anno_dict[:, 4])
    # print(f'cat id {image_anno_dict[:, 4]}')
    # print(f'cls id {gt_label}')
    target = {
        'image_id': image_id,
        'boxes': gt_box,
        'labels': gt_label,
        'ori_size': ori_shape,
        'size': image_shape
    }
    image, target = trans(image, target)
    return out_data(image, target)


def create_detr_dataset(args, mindrecord_file, batch_size=2, device_num=1,
                        rank_id=0, is_training=True, num_parallel_workers=8, python_multiprocessing=False):
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)
    ds = de.MindDataset(mindrecord_file, columns_list=["image_id", "image", "annotation"], num_shards=device_num,
                        shard_id=rank_id, num_parallel_workers=num_parallel_workers, shuffle=is_training)
    decode = C.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = (lambda image_id, image, annotation: preprocess_fn(args, image_id, image, annotation, is_training))

    if is_training:
        ds = ds.map(input_columns=["image_id", "image", "annotation"],
                    output_columns=["image", "mask", "boxes", "labels", "valid", "dn_valid"],
                    column_order=["image", "mask", "boxes", "labels", "valid", "dn_valid"],
                    operations=compose_map_func, python_multiprocessing=python_multiprocessing,
                    num_parallel_workers=num_parallel_workers)
        # ds.project(["image", "mask", "boxes", "labels", "valid"])
        ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image_id", "image", "annotation"],
                    output_columns=["image", "mask", "image_id", "ori_size"],
                    column_order=["image", "mask", "image_id", "ori_size"],
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        # ds.project(["image", "mask", "image_id", "ori_size"])
        ds = ds.batch(batch_size, drop_remainder=False)
    return ds
