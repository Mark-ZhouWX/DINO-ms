import os
import platform

import cv2
import mindspore as ms
from mindspore import nn, ops, Tensor
from common.detr.backbone.resnet import resnet50
from common.detr.matcher.matcher import HungarianMatcher
from common.detr.neck.channel_mapper import ChannelMapper
from common.layers.position_embedding import PositionEmbeddingSine
from common.utils.box_ops import box_xyxy_to_cxcywh
from common.utils.preprocessing import pad_as_batch
from model_zoo.dino.dino import DINO
from model_zoo.dino.dino_transformer import DINOTransformer, DINOTransformerEncoder, DINOTransformerDecoder
from model_zoo.dino.dn_criterion import DINOCriterion
from test import is_windows

# set context
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU' if is_windows else 'GPU')

num_classes = 80
num_queries = 900
# build model
backbone = resnet50(
    in_channels=3,
    norm='FrozenBN',
    out_layers=["res3", "res4", "res5"]
)

position_embedding = PositionEmbeddingSine(
    num_pos_feats=128,
    temperature=10000,
    normalize=True,
    offset=-0.5
)

neck = ChannelMapper(
    input_channels={"res3": 512, "res4": 1024, "res5": 2048},
    in_features=["res3", "res4", "res5"],
    out_channels=256,
    num_outs=4,
    kernel_size=1,
    norm_layer=nn.GroupNorm(num_groups=32, num_channels=256)
)

transformer = DINOTransformer(
    encoder=DINOTransformerEncoder(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        num_layers=6,
        post_norm=False,
        num_feature_levels=4
    ),
    decoder=DINOTransformerDecoder(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        num_layers=6,
        return_intermediate=True,
        num_feature_levels=4,
    ),
    num_feature_levels=4,
    two_stage_num_proposals=num_queries,
)

criterion = DINOCriterion(
    num_classes=num_classes,
    matcher=HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_class_type='focal_loss_cost',
        alpha=0.25,
        gamma=2.0
    ),
    weight_dict=dict(
        loss_class=1,
        loss_bbox=5.0,
        loss_giou=2.0,
        loss_class_dn=1,
        loss_bbox_dn=5.0,
        loss_giou_dn=2.0
    ),
    loss_class_type='focal_loss',
    alpha=0.25,
    gamma=2.0,
    two_stage_binary_cls=False,
)
dino = DINO(backbone,
            position_embedding,
            neck,
            transformer,
            embed_dim=256,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=True,
            criterion=criterion,
            pixel_mean=[123.675, 116.280, 103.530],
            pixel_std=[58.395, 57.120, 57.375],
            select_box_nums_for_evaluation=300,
            dn_number=100,
            label_noise_ratio=0.2,
            box_noise_scale=1.0,
            )

# test inference runtime
image_root = r"C:\02Data\demo\image" if is_windows else '/data1/zhouwuxing/demo/'
image_path1 = os.path.join(image_root, 'hrnet_demo.jpg')
image_path2 = os.path.join(image_root, 'road554.png')
image_path3 = os.path.join(image_root, 'orange_71.jpg')

inputs = [dict(image=Tensor.from_numpy(cv2.imread(image_path1)).transpose(2, 0, 1),
               instances=dict(image_size=(423, 359), gt_classes=Tensor([3, 7], ms.int32),
                              gt_boxes=Tensor([[100, 200, 210, 300], [50, 100, 90, 150]]))),
          dict(image=Tensor.from_numpy(cv2.imread(image_path2)).transpose(2, 0, 1),
               instances=dict(image_size=(400, 300), gt_classes=Tensor([21, 45, 9], ms.int32),
                              gt_boxes=Tensor([[80, 220, 150, 320], [180, 100, 300, 200], [150, 150, 180, 180]]))),
          # dict(image=Tensor.from_numpy(cv2.imread(image_path3)).transpose(2, 0, 1),
          #      instances=dict(image_size=(1249, 1400), gt_classes=Tensor([3, 7]),
          #                     gt_boxes=Tensor([[100, 200, 210, 300], [50, 100, 90, 150]]))),
          ]


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
    train = True
    infer = False

    pth_dir = r"C:\02Data\models" if is_windows else '/data1/zhouwuxing/pretrained_model/'
    pth_path = os.path.join(pth_dir, "dino_r50_4scale_12ep_49_2AP.pth")
    ms_pth_path = os.path.join(pth_dir, "ms_dino_r50_4scale_12ep_49_2AP.ckpt")

    ms.load_checkpoint(ms_pth_path, dino)

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
