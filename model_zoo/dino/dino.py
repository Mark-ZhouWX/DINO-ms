import copy
import math
from typing import List, Optional

import cv2
import mindspore as ms
import numpy as np
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init
import mindspore.numpy as ms_np

from common.layers.mlp import MLP
from common.utils.misc import inverse_sigmoid
from common.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from common.utils.postprocessing import detector_postprocess
from common.utils.preprocessing import pad_as_batch
from common.utils.torch_converter import init_like_torch


class DINO(nn.Cell):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
    """

    def __init__(self,
                 backbone: nn.Cell,
                 position_embedding: nn.Cell,
                 neck: nn.Cell,
                 transformer: nn.Cell,
                 embed_dim: int,
                 num_classes: int,
                 num_queries: int,
                 aux_loss: bool = True,
                 select_box_nums_for_evaluation: int = 300,
                 dn_number: int = 100,
                 label_noise_ratio: float = 0.2,
                 box_noise_scale: float = 1.0,
                 input_format: Optional[str] = "RGB",
                 vis_period: int = 0,
                 unit_test=False,
                 ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Dense(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # define where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss

        # de-noising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        init_like_torch(self.class_embed)
        self.class_embed.bias.set_data(ops.ones(num_classes, self.class_embed.bias.dtype) * bias_value)
        self.bbox_embed.layers[-1].weight.set_data(init.initializer(init.Constant(0),
                                                                    self.bbox_embed.layers[-1].weight.shape,
                                                                    self.bbox_embed.layers[-1].weight.dtype))
        self.bbox_embed.layers[-1].bias.set_data(init.initializer(init.Constant(0),
                                                                  self.bbox_embed.layers[-1].bias.shape,
                                                                  self.bbox_embed.layers[-1].bias.dtype))
        # class weight/ bbox the first n-1 layer
        for neck_layer in self.neck.name_cells():
            if isinstance(neck_layer, nn.Conv2d):
                neck_layer.weight.set_data(init.initializer(init.XavierUniform(gain=1)),
                                           neck_layer.weight.shape, neck_layer.weight.dtype)
                neck_layer.bias.set_data(init.initializer(init.Constant(0),
                                                          neck_layer.bias.shape, neck_layer.bias.dtype))

        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.CellList([copy.deepcopy(self.class_embed) for _ in range(num_pred)])
        self.bbox_embed = nn.CellList([copy.deepcopy(self.bbox_embed) for _ in range(num_pred)])

        bias_init_data = self.bbox_embed[0].layers[-1].bias.data
        bias_init_data[2:] = Tensor(-2.0)
        # p_type, d_type = self.bbox_embed[0].layers[-1].bias.shape, self.bbox_embed[0].layers[-1].bias.dtype
        # self.bbox_embed[0].layers[-1].bias.set_data(init.initializer(bias_init_data, p_type, d_type))

        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        for bbox_embed_layer in self.bbox_embed:
            bias_init_data = bbox_embed_layer.layers[-1].bias.data
            bias_init_data[2:] = Tensor(-0.0)
            p_type, d_type = bbox_embed_layer.layers[-1].bias.shape, bbox_embed_layer.layers[-1].bias.dtype
            bbox_embed_layer.layers[-1].bias.set_data(init.initializer(bias_init_data, p_type, d_type))

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        self.vis_iter = 0
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization"

        # operator
        self.uniform_real = ops.UniformReal()
        self.uniform_int = ops.UniformInt()

        self.unit_test = unit_test

    def construct(self, images, img_masks, targets=None):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            images (Tensor[b, c, h, w]): batch image
            img_masks (Tensor(b, h, w)): image masks with value 1 for padding area and 0 for valid area
        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not self.unit_test:
            batch_size, _, h, w = images.shape
            # extract features with backbone
            features = self.backbone(images)
        else:  # test inference without backbone
            # npz_file = np.load('/data1/zhouwuxing/demo/resnet_fm.npz')
            # features = dict()
            # for k in npz_file.files:
            #     features[k] = Tensor(npz_file[k], dtype=images.dtype)
            features = dict(
                res3=ops.ones((2, 512, 53, 45), ms.float32),
                res4=ops.ones((2, 1024, 27, 23), ms.float32),
                res5=ops.ones((2, 2048, 14, 12), ms.float32)
            )
            # img_masks = ops.zeros((2, 423, 359), ms.float32)
            unpad_img_sizes = Tensor([(423, 359), (400, 300)])
            # targets = self.prepare_targets(args[0], args[1], args[2])

        # model_zoo backbone features to the embed dimension of transformer
        multi_level_feats = self.neck(features)  # list[b, embed_dim, h, w], len=num_level
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            resize_nearest = ops.ResizeNearestNeighbor(size=feat.shape[-2:])
            l_mask = ops.squeeze(resize_nearest(ops.expand_dims(img_masks, 0)), 0)
            l_mask = ops.cast(l_mask, ms.bool_)
            multi_level_masks.append(l_mask)
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # de-noising preprocessing
        # prepare label query embeding
        if self.training:
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            # inference does not need dn
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (inter_states, init_reference, inter_reference, enc_state, enc_reference) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,  # gt query and target
            attn_masks=[attn_mask, None],
        )

        # hack implementation for distributed training
        inter_states[0] += self.label_enc.embedding_table[0, 0] * 0.0

        # calculate output coordinates and classes
        outputs_classes = []
        outputs_coords = []

        for lvl in range(inter_states.shape[0]):
            reference = init_reference if lvl == 0 else inter_reference[lvl - 1]
            reference = inverse_sigmoid(reference)
            l_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                # for anchor contains only x,y
                assert reference.shape[-1] == 2
                tmp[..., 2] += reference
            l_coord = tmp.sigmoid()
            outputs_classes.append(l_class)
            outputs_coords.append(l_coord)
        # [num_decoder_layers, bs, num_query, num_classes]
        outputs_class = ops.stack(outputs_classes)
        # [num_decoder_layers, bs, num_query, 4]
        outputs_coord = ops.stack(outputs_coords)

        # de-noising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_meta)

        # output (tuple(tuple(Tensor))) with size 3 -> last, aux, two_stage
        output = [None, None, None]

        # prepare for loss computation
        output[0] = (outputs_class[-1], outputs_coord[-1])
        if self.aux_loss:
            # len(output["aux_outputs"]) = num_decoder_layer - 1
            output[1] = (outputs_class[:-1], outputs_coord[:-1])

        # prepare two stage output
        interm_coord = enc_reference
        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        # TODO 感觉这里可以写在transformer class里，直接输出interm_class，不知作者这么写是不是为了兼容其他的detr
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output[2] = (interm_class, interm_coord)

        return output

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        """fill output_known bboxes and logits in dn_meta, separate predictions of gt and matching part"""
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                # output of the layers before the last one
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def prepare_for_cdn(
            self,
            targets,
            dn_number,
            label_noise_ratio,
            box_noise_scale,
            num_queries,
            num_classes,
            hidden_dim,
            label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little.
        Args:
            dn_number: total number of positive gt queries
            num_queries: number of quires
            num_classes: number of classes
            hidden_dim: transformer hidden dim
            label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
        dn_number = dn_number * 2  # positive and negative
        # [labels_0, labels_1, ...] len(label_i)=num_instance of image_i
        # eg: [(1,1,1), (1,1), (1,1,1,1)], batch_size=3, num_instance in each image is 3,2,4
        known = [(ops.ones_like(t["labels"])) for t in targets]  # len(known)=batch_size
        batch_size = len(known)
        # num_instance of each image
        # eg: [3, 2, 4]
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        # gt_query group num, each group has the length of max_num_instance
        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        # flattened labels, (sum(instance_i))
        # eg: [7,3,2, 6,8, 9,1,4,5]
        labels = ops.concat([t["labels"] for t in targets])
        # flattened boxes (sum(instance_i), 4)
        boxes = ops.concat([t["boxes"] for t in targets])
        # flattened batch_id
        # [0,0,0, 1,1, 2,2,2,2]
        batch_idx = ops.concat(
            [ms_np.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        # (sum(instance_i) * 2 * dn_number)
        known_labels = ms_np.tile(labels, (2 * dn_number, 1)).view(-1)
        # (sum(instance_i) * 2 * dn_number)
        # eg: [0,0,0, 1,1, 2,2,2,2,  0,0,0, 1,1, 2,2,2,2, ...]
        known_bid = ms_np.tile(batch_idx, (2 * dn_number, 1)).view(-1)
        # (sum(instance_i)* 2 * dn_number, 4)
        known_bboxs = ms_np.tile(boxes, (2 * dn_number, 1))
        known_labels_expaned = copy.deepcopy(known_labels)
        known_bbox_expand = copy.deepcopy(known_bboxs)

        if label_noise_ratio > 0:
            # (sum(instance_i) * 2 * dn_number)
            p = self.uniform_real(known_labels_expaned.shape)  # uniform(0,1)
            # indice of lower p,
            chosen_indice = ops.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = self.uniform_int(
                chosen_indice.shape, Tensor(0, dtype=ms.int32), Tensor(num_classes, dtype=ms.int32)
            )  # randomly put a new label_id here
            # new_label = ops.cast(new_label, ms.int64)
            ops.tensor_scatter_elements(known_labels_expaned, indices=chosen_indice, updates=new_label, axis=0)
        single_padding = int(max(known_num))

        # equal to original dn_number
        pad_size = int(single_padding * 2 * dn_number)
        # (dn_number_group, sum(instance))   sum(instance): total number of gt boxes in a batch
        positive_idx = ms_np.tile(ops.arange(end=len(boxes), dtype=ms.int64).unsqueeze(0), (dn_number, 1))
        # (dn_number, 1) -> (dn_number_group, sum(instance))
        positive_idx += (ops.arange(end=dn_number, dtype=ms.int64) * len(boxes) * 2).unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = ops.zeros_like(known_bboxs)
            # left bottom and right top points of box
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            # copy of half wh
            diff = ops.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            # random 1 or -1
            rand_sign = ops.cast((self.uniform_int(
                known_bboxs.shape, Tensor(0, ms.int32), Tensor(2, ms.int32)) * 2.0 - 1.0), ms.float32)
            rand_part = self.uniform_real(known_bboxs.shape)  # uniform(0,1)

            # negative bbox has offset within (1, 2)*half_wh, positive has (0, 1)*half_wh
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + ops.mul(rand_part, diff) * box_noise_scale
            known_bbox_ = ops.clip_by_value(known_bbox_, clip_value_min=0.0, clip_value_max=1.0)  # prevent out of image

            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2  # gaussion distribution
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long()
        # (sum(instance_i) * 2 * dn_number, embed_dim)
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = ms_np.zeros((pad_size, hidden_dim))
        padding_bbox = ms_np.zeros((pad_size, 4))

        input_query_label = ms_np.tile(padding_label, (batch_size, 1, 1))
        input_query_bbox = ms_np.tile(padding_bbox, (batch_size, 1, 1))

        map_known_indice = ops.Tensor([])
        if len(known_num):
            map_known_indice = ops.concat(
                [ops.arange(end=num) for num in known_num]
            )  # eg:[0,1,2, 0,1, 0,1,2,3]
            # eg: [[0,1,2, 0,1, 0,1,2,3,    4,5,6, 4,5, 4,5,6,7,      8,9,10, 8,9, 8,9,10,11], ...] single_padding=4
            map_known_indice = ops.concat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = ops.ones((tgt_size, tgt_size), ms.float32) < 0  # all false, means no mask
        # match query cannot see gt query
        attn_mask[pad_size:, :pad_size] = True  # left bottom gray part of the figure in the paper
        # gt cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                single_padding * 2 * i: single_padding * 2 * (i + 1),
                single_padding * 2 * (i + 1): pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[single_padding * 2 * i: single_padding * 2 * (i + 1), : single_padding * i * 2] = True
            else:
                # gt queries after the i-th gt query
                attn_mask[
                single_padding * 2 * i: single_padding * 2 * (i + 1),
                single_padding * 2 * (i + 1): pad_size,
                ] = True
                # # gt queries before the i-th gt query
                attn_mask[
                single_padding * 2 * i: single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,  # query num per group, p+n
            "dn_num": dn_number,  # group num
        }
        # (bs, original dn_number * 2, embed_dim)
        # (bs, original dn_number * 2, 4)
        # (bs, original dn_number * 2, original dn_number * 2)
        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def preprocess_image(self, batched_inputs):
        images = batched_inputs['image']
        img_masks = batched_inputs['mask']
        gt_bboxes = batched_inputs['boxes']
        gt_labels = batched_inputs['labels']
        gt_masks = batched_inputs['valid']
        return images, img_masks, gt_bboxes, gt_labels, gt_masks

    def prepare_targets(self, gt_bboxes, gt_labels, gt_valids):
        new_targets = []
        bs = len(gt_labels)
        for i in range(bs):
            new_targets.append({"labels": ops.masked_select(gt_labels[i], gt_valids[i]),
                                "boxes": ops.masked_select(gt_bboxes[i], gt_valids[i][:, None]).reshape(-1, 4)})

        return new_targets

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
