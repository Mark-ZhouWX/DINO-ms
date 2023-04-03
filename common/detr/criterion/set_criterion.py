from typing import List

import mindspore as ms
import mindspore.numpy as ms_np
from mindspore import nn, ops, Tensor

from common.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = ops.sigmoid(inputs)
    _, _, num_class = inputs.shape
    weight = ops.ones(num_class, inputs.dtype)
    pos_weight = ops.ones(num_class, inputs.dtype)
    ce_loss = ops.binary_cross_entropy_with_logits(inputs, targets,
                                                   weight=weight, pos_weight=pos_weight, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Cell):
    """
    This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            num_classes,
            matcher,
            weight_dict,
            losses: List[str] = ["class", "boxes"],
            eos_coef: float = 0.1,
            loss_class_type: str = "focal_loss",
            alpha: float = 0.25,
            gamma: float = 2.0,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.eos_coef = eos_coef
        self.loss_class_type = loss_class_type
        assert loss_class_type in [
            "ce_loss",
            "focal_loss",
        ], "only support ce loss and focal loss for computing classification loss"

        if self.loss_class_type == "ce_loss":
            empty_weight = ms.Parameter(ops.ones(self.num_classes + 1, ms.float32), requires_grad=False)
            empty_weight[-1] = eos_coef
            # self.register_buffer("empty_weight", empty_weight)
            self.empty_weight = empty_weight
        self.l1_loss = nn.L1Loss(reduction="none")

    def loss_labels(self, outputs, targets, indices):
        """
        Classification loss (Binary focal loss)
        outputs (Tuple[Tensor]): predictions, contains logits and bbox.
        targets (dict): targets, a dict that contains the key "labels" containing a tensor of dim [nb_target_boxes]
        indices (List[Tuple[Tensor, Tensor]]): list with length batch_size, each element
                                                is the indices of source query and target bbox.
        """

        src_logits, _ = outputs  # (bs, num_query, num_class),
        tgt_labels, _, tgt_valids = targets
        tgt_labels *= tgt_valids.astype(ms.int32)
        tgt_labels += ops.logical_not(tgt_valids).astype(ms.int32) * self.num_classes  # replace unvalid with num_class

        bs, num_pad_box = tgt_valids.shape
        num_valid_box = ops.reduce_sum(tgt_valids.astype(ms.float32))

        _, num_query, num_class = src_logits.shape

        # Assign gt label to all queries, the label of 'no object' is num_class
        src_ind, tgt_ind = indices  # (bs, num_pad_box),  (bs, num_pad_box)
        src_ind = (src_ind + num_query) % num_query  # replace -1 with num_query-1
        tgt_ind = (tgt_ind + num_pad_box) % num_pad_box  # replace -1 with num_pad_box-1

        # (bs, num_query)  value=80
        target_classes = ms_np.full((bs, num_query), self.num_classes, dtype=ms.float32)
        # for i in range(bs):
        #     one_tc =ops.gather(tgt_labels[i].astype(ms.float32), tgt_ind[i], 0)  # (bs, num_pad_box)
        #     target_classes[i] = ops.scatter_update(target_classes[i], src_ind[i], one_tc)
        tc = ops.gather_elements(tgt_labels.astype(ms.float32), 1, tgt_ind)
        target_classes = ops.tensor_scatter_elements(target_classes, src_ind, tc, 1)
        target_classes = target_classes.astype(ms.int32)

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            # TODO to check transpose, why
            loss_class = ops.cross_entropy(ops.transpose(src_logits, (0, 2, 1)), target_classes, self.empty_weight)
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = ops.zeros((bs, num_query, num_class + 1), ms.float32)
            ones = ops.ones_like(ops.expand_dims(target_classes, -1)).astype(ms.float32)
            target_classes_onehot = ops.tensor_scatter_elements(target_classes_onehot,
                                                                ops.expand_dims(target_classes, -1), ones, axis=2)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            # TODO better use this
            # target_classes_onehot = ops.one_hot(target_classes,
            #                                     depth=num_class + 1, on_value=Tensor(1), off_value=Tensor(0))[:, :, :-1]

            loss_class = (
                    sigmoid_focal_loss(
                        src_logits,
                        target_classes_onehot,
                        num_boxes=num_valid_box,
                        alpha=self.alpha,
                        gamma=self.gamma,
                    )
                    * src_logits.shape[1]
            )
        else:
            raise NotImplementedError(f'support only ce_loss and focal_loss, but got {self.loss_class_type}')

        losses = {"loss_class": loss_class}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        _, src_boxes = outputs
        _, tgt_boxes, tgt_valids = targets
        src_ind, tgt_ind = indices  # (bs, num_pad_box),  (bs, num_pad_box)

        num_valid_box = ops.reduce_sum(tgt_valids.astype(ms.float32))
        bs, num_query, _= src_boxes.shape
        _, num_pad_box = tgt_valids.shape

        src_ind = (src_ind + num_query) % num_query  # replace -1 with num_query-1
        tgt_ind = (tgt_ind + num_pad_box) % num_pad_box  # replace -1 with num_pad_box-1

        # gather operator does not have batch operation
        sorted_src_boxes = ops.zeros_like(tgt_boxes)  # (bs, num_pad_box, 4)
        sorted_tgt_boxes = ops.zeros_like(tgt_boxes)  # (bs, num_pad_box, 4)
        for i in range(bs):
            sorted_src_boxes[i] = ops.gather(src_boxes[i], src_ind[i], 0)
            sorted_tgt_boxes[i] = ops.gather(tgt_boxes[i], tgt_ind[i], 0)

        losses = {}
        loss_bbox = self.l1_loss(sorted_src_boxes, sorted_tgt_boxes)

        loss_bbox *= tgt_valids.astype(ms.float32).reshape(bs, num_pad_box, 1)
        losses["loss_bbox"] = loss_bbox.sum() / num_valid_box

        # (bs, num_pad_box, 4) -> (bs*num_pad_box, 4) -> (bs*num_pad_box, bs*num_pad_box) -> (bs*num_pad_box,)
        giou_matrix = generalized_box_iou(box_cxcywh_to_xyxy(sorted_src_boxes.reshape(bs*num_pad_box, 4)),
                                            box_cxcywh_to_xyxy(sorted_tgt_boxes.reshape(bs*num_pad_box, 4)))
        loss_giou = ops.sub(1, giou_matrix.diagonal())
        loss_giou *= tgt_valids.astype(ms.float32).reshape(bs*num_pad_box)

        # loss_giou_list = []
        # for i in range(bs):
        #     a = 1 - generalized_box_iou(box_cxcywh_to_xyxy(sorted_src_boxes[i].reshape(num_pad_box, 4)),
        #                         box_cxcywh_to_xyxy(sorted_tgt_boxes[i].reshape(num_pad_box, 4))).diagonal()
        #     loss_giou_list.append(a)
        # loss_giou = ops.stack(loss_giou_list)  # (bs, num_pad_box)
        # loss_giou *= tgt_valids.astype(ms.float32)

        losses["loss_giou"] = loss_giou.sum() / num_valid_box

        return losses

    # FIXME @ms.ms_function 1.dict has no update  2. cannot return dict
    def get_loss(self, outputs, targets, indices):
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))

        return losses

    def construct(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        # TODO to finsh static part
        raise NotImplementedError('not realize static part yet')

        return_indices = False
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(t["labels"].shape[0] for t in targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # to replace with outputs_without_aux
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def compute_weighted_loss(self, loss_dict):
        for k in loss_dict.keys():
            if k in self.weight_dict:
                loss_dict[k] *= self.weight_dict[k]
        loss = sum(loss_dict.values())
        return loss

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "loss_class_type: {}".format(self.loss_class_type),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "focal loss alpha: {}".format(self.alpha),
            "focal loss gamma: {}".format(self.gamma),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
