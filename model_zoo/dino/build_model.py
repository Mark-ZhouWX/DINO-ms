from mindspore import nn

from common.detr.backbone.resnet import resnet50
from common.detr.matcher.matcher import HungarianMatcher
from common.detr.neck.channel_mapper import ChannelMapper
from common.layers.position_embedding import PositionEmbeddingSine
from model_zoo.dino.dino import DINO
from model_zoo.dino.dino_transformer import DINOTransformer, DINOTransformerEncoder, DINOTransformerDecoder
from model_zoo.dino.dn_criterion import DINOCriterion


def build_dino(unit_test=False):
    num_classes = 80
    num_queries = 900
    # dn_number = 0 if unit_test else 100
    dn_number = 0
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
        embed_dim=256,
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
                select_box_nums_for_evaluation=300,
                dn_number=dn_number,
                label_noise_ratio=0.2,
                box_noise_scale=1.0,
                unit_test=unit_test,
                )

    return dino, criterion

