import os
import time
from typing import Dict

import cv2
import mindspore as ms
import numpy as np
from mindspore import ops, Tensor, set_seed
import mindspore.numpy as ms_np
from pycocotools.coco import COCO
from tqdm import tqdm

from common.dataset.coco_eval import CocoEvaluator
from common.dataset.dataset import create_mindrecord, create_detr_dataset, coco_classes, coco_clsid_to_catid, \
    coco_id_dict
from common.utils.box_ops import box_cxcywh_to_xyxy, box_scale
from common.utils.system import is_windows
from config import config
from model_zoo.dino.build_model import build_dino
from test.dino import get_input, convert_input_format_with_resizepad


def select_from_prediction(box_cls, box_pred, num_select=300):

    bs, num_query, num_class = box_cls.shape

    # box_cls.shape: 1, 300, 80
    # box_pred.shape: 1, 300, 4
    prob = box_cls.sigmoid()
    # TODO 不理解为什么如此选取topk，这样选取的bbox有许多重复的, 用一个query可以预测多个class
    # num_query*num_class must on the last axis
    # (bs, num_query, num_class) -> (bs, num_query*num_class) -> (bs, num_select) + (bs, num_select)
    topk_values, topk_indexes_full = ops.topk(prob.view(bs, -1), num_select)
    scores = topk_values
    # (bs, num_select)
    topk_boxes_ind = ops.div(topk_indexes_full.astype(ms.float32),
                             num_class, rounding_mode="floor").astype(ms.int32)

    labels = topk_indexes_full % num_class  # (bs, num_select)
    boxes = ops.gather_elements(box_pred, 1, ms_np.tile(topk_boxes_ind.unsqueeze(-1), (1, 1, 4)))  # (bs,num_eval,4)

    return scores, labels, boxes


def inference(model, image, mask, ori_size, num_select=300):
    # image, mask, image_id, ori_size = data
    output = model(image, mask)

    box_cls = output["pred_logits"]
    box_pred = output["pred_boxes"]
    assert len(box_cls) == len(image)
    scores, labels, boxes = select_from_prediction(box_cls, box_pred, num_select)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)  # (bs, num_select, 4)
    boxes_xyxy_scaled = box_scale(boxes_xyxy, scale=ori_size)  # (bs, num_select, 4)
    return scores, labels, boxes_xyxy_scaled


def visualize(pred_dict: Dict, coco_gt: COCO, save_dir, raw_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_id, res in pred_dict.items():
        img_file_info = coco_gt.loadImgs(img_id)[0]
        save_path = os.path.join(save_dir, img_file_info['file_name'])
        raw_path = os.path.join(raw_dir, img_file_info['file_name'])
        choose = res['scores'] > 0.3
        labels = ops.masked_select(res['labels'], choose).asnumpy()
        boxes = ops.masked_select(res['boxes'], choose.unsqueeze(-1)).asnumpy().reshape(-1, 4)
        scores = ops.masked_select(res['scores'], choose).asnumpy()
        image = cv2.imread(raw_path)

        for s, l, b in zip(scores, labels, boxes):
            x1, y1, x2, y2 = b
            class_name = coco_id_dict[l]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            cat_id = ann['category_id']
            class_name = coco_gt.cats[cat_id]['name']
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(image, class_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(save_path, image)


def coco_evaluate(model, eval_dateset, eval_anno_path, save_dir, raw_dir, save_vis=False):
    # coco evaluator
    coco_gt = COCO(eval_anno_path)
    coco_evaluator = CocoEvaluator(coco_gt, ('bbox', ))

    # inference
    start_time = time.time()
    num_select = 300
    ds_size = dataset.get_dataset_size()
    iii = 0
    for data in tqdm(eval_dateset.create_dict_iterator(), total=ds_size, desc=f'inferring...'):
        image_id = data['image_id'].asnumpy()  # (bs, )
        image = data['image']  # (bs, c, h, w)
        mask = data['mask']  # (bs, h, w)
        size_wh = data['ori_size'][:, ::-1]  # (bs, 2), in wh order
        scores, labels, boxes = inference(model, image, mask, size_wh, num_select)
        cat_ids = Tensor(np.vectorize(coco_clsid_to_catid.get)(labels.asnumpy()))
        res = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, cat_ids, boxes)]
        img_res = {int(idx): output for idx, output in zip(image_id, res)}
        coco_evaluator.update(img_res)
        if save_vis:
            visualize(img_res, coco_gt, save_dir, raw_dir)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    print(coco_evaluator.coco_eval.get('bbox').stats)
    print(f'cost time: {time.time() - start_time}s', )
    print("\n========================================\n")


def box_cxywh2xyxy_and_scale(box, scale):
    b_cxywh = box_cxcywh_to_xyxy(box)
    b_scaled = box_scale(b_cxywh, scale)
    return b_scaled


def evaluate_single(model):
    raw_inputs, image_root = get_input()
    save_dir = os.path.join(image_root, 'demo_vis')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    images, img_masks, gt_classes_list, gt_boxes_list, gt_valids_list, ori_size = convert_input_format_with_resizepad(raw_inputs)
    size_wh = ori_size[:, ::-1]
    print(f'scale size  ', size_wh)
    scores_batch, labels_batch, boxes_batch = inference(model, images, img_masks, size_wh, 300)
    pred_list = [{'scores':s, 'labels':l, 'boxes':b} for s, l, b in zip(scores_batch, labels_batch, boxes_batch)]

    gt_list = [{'labels':l, 'boxes':box_cxywh2xyxy_and_scale(b, im_s)}
               for l, b, im_s in zip(gt_classes_list, gt_boxes_list, size_wh)]
    for ii, (ipt, res, gt) in enumerate(zip(raw_inputs, pred_list, gt_list)):
        image = ipt['image'].transpose(1, 2, 0).asnumpy()
        choose = res['scores'] > 0.4
        if choose.any():
            labels = ops.masked_select(res['labels'], choose).asnumpy()
            boxes = ops.masked_select(res['boxes'], choose.unsqueeze(-1)).reshape(-1, 4).asnumpy()
            scores = ops.masked_select(res['scores'], choose).asnumpy()
        else:
            scores, labels, boxes = [], [], []

        for s, l, b in zip(scores, labels, boxes):
            x1, y1, x2, y2 = b
            class_name = coco_classes[l]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for l, b in zip(gt['labels'], gt['boxes']):
            x1, y1, x2, y2 = b
            class_name = coco_classes[l]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        save_path = os.path.join(save_dir, f'{ii}.jpg')
        cv2.imwrite(save_path, image)



if __name__ == '__main__':
    # set context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU' if is_windows else 'GPU',
                   pynative_synchronize=False)
    rank = 0
    device_num = 1
    set_seed(0)
    eval_model = build_dino()
    eval_model.set_train(False)

    pth_dir = './work_dirs/'
    model_path = os.path.join(pth_dir, "dino_epoch009_rank0.ckpt")
    print(f'load model from {model_path}')
    ms.load_checkpoint(model_path, eval_model)

    evaluate_coco = True
    if evaluate_coco:
        # evaluate coco
        mindrecord_file = create_mindrecord(config, rank, "DETR.mindrecord.eval", False)
        dataset = create_detr_dataset(config, mindrecord_file, batch_size=1,
                                      device_num=device_num, rank_id=rank,
                                      # num_parallel_workers=config.num_parallel_workers,
                                      num_parallel_workers=1,
                                      python_multiprocessing=config.python_multiprocessing,
                                      is_training=False)

        anno_json = os.path.join(config.coco_path, "annotations/instances_val2017.json")
        vis_save_dir = os.path.join(config.coco_path, 'val2017_vis')
        raw_img_dir = os.path.join(config.coco_path, 'val2017')
        coco_evaluate(eval_model, dataset, anno_json, vis_save_dir, raw_img_dir)
    else:
        evaluate_single(eval_model)


