from typing import List

import mindspore as ms
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
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
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
            empty_weight = ops.ones(self.num_classes + 1)
            empty_weight[-1] = eos_coef
            # self.register_buffer("empty_weight", empty_weight)

            # TODO to check whether this is ok
            self.empty_weight = empty_weight

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # (bs, num_query, num_class)
        bs, num_query, num_class = src_logits.shape

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = ops.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = ops.full(src_logits.shape[:2], self.num_classes, dtype=ms.int64)
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            loss_class = ops.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = ops.zeros(
                [bs, num_query, num_class + 1], dtype=src_logits.dtype)

            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]

            # TODO better use this
            # target_classes_onehot = ops.one_hot(target_classes,
            #                                     depth=num_class + 1, on_value=Tensor(1), off_value=Tensor(0))[:, :, :-1]

            loss_class = (
                    sigmoid_focal_loss(
                        src_logits,
                        target_classes_onehot,
                        num_boxes=num_boxes,
                        alpha=self.alpha,
                        gamma=self.gamma,
                    )
                    * src_logits.shape[1]
            )
        else:
            raise NotImplementedError(f'support only ce_loss and focal_loss, but got {self.loss_class_type}')

        losses = {"loss_class": loss_class}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = ops.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = ops.functional.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}

        loss_giou = 1 - ops.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = ops.cat([ops.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = ops.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = ops.cat([ops.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = ops.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = Tensor([num_boxes], dtype=ms.float32, )
        # TODO multi-node distribution, not fully tested 2
        if ms.communication.GlobalComm.INITED:
            ops.AllReduce(num_boxes)

        group_size = ms.communication.get_group_size() if ms.communication.GlobalComm.INITED else 1
        num_boxes = ops.clamp(num_boxes / group_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
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
