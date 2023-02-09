import os
import platform

import cv2
import mindspore as ms
from mindspore import nn, ops, Tensor
from common.detr.backbone.resnet import resnet50
from common.detr.matcher.matcher import HungarianMatcher
from common.detr.neck.channel_mapper import ChannelMapper
from common.layers.position_embedding import PositionEmbeddingSine
from model_zoo.dino.dino import DINO
from model_zoo.dino.dino_transformer import DINOTransformer, DINOTransformerEncoder, DINOTransformerDecoder
from model_zoo.dino.dn_criterion import DINOCriterion
from test import is_windows

# set context
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

# build model
backbone = resnet50(
    in_channels=3,
    norm=nn.BatchNorm2d,
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
    two_stage_num_proposals=900,
)

criterion = DINOCriterion(
    num_classes=80,
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
            num_classes=80,
            num_queries=900,
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
image_root = r"C:\02Data\demo\image" if is_windows else '/data/zhouwuxing/demo/'
image_path1 = os.path.join(image_root, 'hrnet_demo.jpg')
image_path2 = os.path.join(image_root, 'road554.png')

inputs = [dict(image=Tensor.from_numpy(cv2.imread(image_path1)).transpose(2, 0, 1),
               instances=dict(image_size=(423, 359), gt_classes=Tensor([3, 7]),
                              gt_boxes=Tensor([[100, 200, 210, 300], [50, 100, 90, 150]]))),
          dict(image=Tensor.from_numpy(cv2.imread(image_path2)).transpose(2, 0, 1),
               instances=dict(image_size=(400, 300), gt_classes=Tensor([21, 45, 9]),
                              gt_boxes=Tensor([[80, 220, 150, 320], [180, 100, 300, 200], [150, 150, 180, 180]])))]


if __name__ == "__main__":
    train = False
    infer = True
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
        dino.set_train()
        loss_dict = dino(inputs)
        print(loss_dict)

    # train one step
    pass
