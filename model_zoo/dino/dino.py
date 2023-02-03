import copy
import math
from typing import List, Optional

import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init

from common.layers.mlp import MLP
from common.utils.misc import inverse_sigmoid
from common.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from common.utils.postprocessing import detector_postprocess


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
        device (str): Training device. Default: "cuda".
    """

    def __init__(self,
                 backbone: nn.Cell,
                 position_embedding: nn.Cell,
                 neck: nn.Cell,
                 transformer: nn.Cell,
                 embed_dim: int,
                 num_classes: int,
                 num_queries: int,
                 criterion: nn.Cell,
                 pixel_mean: List[float] = [123.675, 116.280, 103.530],
                 pixel_std: List[float] = [58.395, 57.120, 57.375],
                 aux_loss: bool = True,
                 select_box_nums_for_evaluation: int = 300,
                 device="cuda",
                 dn_number: int = 100,
                 label_noise_ratio: float = 0.2,
                 box_noise_scale: float = 1.0,
                 input_format: Optional[str] = "RGB",
                 vis_period: int = 0,
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

        # define where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # de-noising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = Tensor(pixel_mean).view(3, 1, 1)
        pixel_std = Tensor(pixel_std).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.set_data(ops.ones(num_classes) * bias_value)
        bbox_last_layer_weight = self.bbox_embed.layers[-1].weight
        bbox_last_layer_weight.set_data(init.initializer(init.Constant(0),
                                                         bbox_last_layer_weight.shape, bbox_last_layer_weight.dtype))
        for _, neck_layer in self.neck.name_cells():
            if isinstance(neck_layer, nn.Conv2d):
                neck_layer.weight.set_data(init.initializer(init.XavierUniform(gain=1)),
                                           neck_layer.weight.shape, neck_layer.weight.dtype)
                neck_layer.bias.set_data(init.initializer(init.Constant(0),
                                                          neck_layer.bias.shape, neck_layer.bias.dtype))

        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.CellList([copy.deepcopy(self.class_embed) for _ in range(num_pred)])
        self.bbox_embed = nn.CellList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])

        # TODO bias修改部分数据
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        for bbox_embed_layer in self.bbox_embed:
            # TODO bias修改部分数据
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        self.vis_iter = 0
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization"

    def construct(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta information and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

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
        # pad images in the batch to the same size (max image size in the batch)
        images, unpad_img_sizes = self.preprocess_image(batched_inputs)  # tensor (b, c, h, w)

        # calculate input mask, only training model need mask
        batch_size, _, H, W = images.shape
        img_masks = images.new_ones((batch_size, H, W))
        if self.training:
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, : img_w] = 0

        # extract features with backbone
        features = self.backbone(images)

        # model_zoo backbone features to the embed dimension of transformer
        multi_level_feats = self.neck(features)  # list[b, embed_dim, h, w], len=num_level
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            resize_nearest = ops.ResizeNearestNeighbor(size=feat.shape[-2:])
            multi_level_masks.append(
                ops.squeeze(resize_nearest(ops.expand_dims(img_masks, 0)))
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # de-noising preprocessing
        # prepare label query embeding
        if self.training:
            gt_instances = [x["instances"] for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
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
            targets = None
            # inference does not need dn
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (inter_states, init_reference, inter_reference, enc_state, enc_reference) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,  # gt query and target
            attn_mask=[attn_mask, None],
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
            output_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_meta)

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            # visualize training samples
            if self.vis_period > 0:
                if self.vis_iter % self.vis_period == 0:
                    box_cls = output["pred_logits"]
                    box_pred = output["pred_boxes"]
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
                self.vis_iter += 1

            # TODO compute loss
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, unpad_img_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, unpad_img_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def inference(self, box_cls, box_pred, image_sizes):
        """
                Arguments:
                    box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                        The tensor predicts the classification probability for each query.
                    box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                        The tensor predicts 4-vector (x,y,w,h) box
                        regression values for every queryx
                    image_sizes (List[torch.Size]): the raw input image sizes without padding

                Returns:
                    results (List[Instances]): a list of #images elements.
                """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        # TODO 不理解为什么如此选取topk，这样选取的bbox有许多重复的
        topk_values, topk_indexes = ops.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = ops.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = ops.gather_elements(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
                zip(scores, labels, boxes, image_sizes)
        ):
            result = dict(image_size=image_size)
            # [num_query, 4]
            pred_boxes = box_cxcywh_to_xyxy(box_pred_per_image)
            pred_boxes[:, 0::2] *= image_size[1]
            pred_boxes[:, 1::2] *= image_size[0]
            result.update(pred_boxes=pred_boxes)
            result.update(scores=scores_per_image)
            result.update(pred_classes=labels_per_image)
            results.append(result)
        return results

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
        # flattened mask
        # [1,1,1,  1,1,  1,1,1,1]
        # unmask_bbox = unmask_label = ops.cat(known)
        # flattened labels, (sum(instance_i))
        # eg: [7,3,2, 6,8, 9,1,4,5]
        labels = ops.cat([t["labels"] for t in targets])
        # flattened boxes (sum(instance_i), 4)
        boxes = ops.cat([t["boxes"] for t in targets])
        # flattened batch_id
        # [0,0,0, 1,1, 2,2,2,2]
        batch_idx = ops.cat(
            [ops.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        # # eg: [[0], [1], [2], [3], [4], [5], [6]]
        # known_indice = ops.nonzero(unmask_label + unmask_bbox)
        # # [0, 1, 2, 3, 4, 5, 6]
        # known_indice = known_indice.view(-1)
        #
        # known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)

        # (sum(instance_i) * 2 * dn_number)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        # (sum(instance_i) * 2 * dn_number)
        # eg: [0,0,0, 1,1, 2,2,2,2,  0,0,0, 1,1, 2,2,2,2, ...]
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        # (sum(instance_i)* 2 * dn_number, 4)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            # (sum(instance_i) * 2 * dn_number)
            p = ops.rand_like(known_labels_expaned.float())  # uniform(0,1)
            # indice of lower p,
            chosen_indice = ops.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = ops.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new label_id here
            ops.tensor_scatter_elements(known_labels_expaned, indices=chosen_indice, updates=new_label, axis=0)
        single_padding = int(max(known_num))

        # equal to original dn_number
        pad_size = int(single_padding * 2 * dn_number)
        # (dn_number_group, sum(instance))
        positive_idx = (
            ops.Tensor(range(len(boxes))).long().unsqueeze(0).repeat(dn_number, 1)
        )

        positive_idx += (ops.Tensor(range(dn_number)) * len(boxes) * 2).long().unsqueeze(1)
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
            rand_sign = (
                ops.randint_like(known_bboxs, low=0, high=2, dtype=ms.float32) * 2.0 - 1.0
            )
            rand_part = ops.rand_like(known_bboxs)  # uniform(0,1)
            # negative bbox has offset within (1, 2)*half_wh, positive has (0, 1)*half_wh
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + ops.mul(rand_part, diff) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)  # prevent out of image
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long()
        # (sum(instance_i) * 2 * dn_number, embed_dim)
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = ops.zeros(pad_size, hidden_dim)
        padding_bbox = ops.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = ops.Tensor([])
        if len(known_num):
            map_known_indice = ops.cat(
                [ops.Tensor(range(num)) for num in known_num]
            )  # eg:[0,1,2, 0,1, 0,1,2,3]
            map_known_indice = ops.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()  # eg: [[4,5,6, 4,5, 4,5,6,7], [8,9,10, 8,9, 8,9,10,11], ...] single_padding=4
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = ops.ones((tgt_size, tgt_size)) < 0  # all false, means no mask
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
                attn_mask[
                    single_padding * 2 * i: single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                # gt queries after the i-th gt query
                attn_mask[
                    single_padding * 2 * i: single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1): pad_size,
                ] = True
                # # gt queries before the i-th gt query
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,  # query num per group
            "dn_num": dn_number,  # group num
        }
        # (bs, original dn_number * 2, embed_dim)
        # (bs, original dn_number * 2, 4)
        # (bs, original dn_number * 2, original dn_number * 2)
        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image['image_size']
            image_size_xyxy = Tensor([w, h, w, h], dtype=ms.float32)
            gt_classes = targets_per_image['gt_classes']
            gt_boxes = targets_per_image['gt_boxes'] / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
