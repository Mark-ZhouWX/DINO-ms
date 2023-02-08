import mindspore as ms
import mindspore.numpy as ms_np
from mindspore import Tensor, ops

from common.detr.criterion.set_criterion import SetCriterion


class TwoStageCriterion:
    pass


class DINOCriterion(TwoStageCriterion):
    """
    This class computes the loss for DINO.
    Add dn loss to TwoStageCriterion, two type of loss will be computed:
    1) two stage loss, including normal detr decoder loss and encoder proposal loss
    2) dn and its auxiliary loss
    """

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             dn_metas: de-noising information, including dn predicts and dn_number, etc.
        """
        losses = super(DINOCriterion, self).forward(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)  # total number of instances of a batch
        num_boxes = Tensor([num_boxes], dtype=ms.float32)
        # TODO multi-node distribution, not fully tested
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        if ms.communication.GlobalComm.INITED:
            ops.AllReduce(num_boxes)

        group_size = ms.communication.get_group_size() if ms.communication.GlobalComm.INITED else 1
        num_boxes = ops.clamp(num_boxes / group_size, min=1).item()

        # Compute all the requested losses

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)

        return losses

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            aux_num: aux loss number
            targets (List[Dict]): contains instances of one batch
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    # eg [0, 1, 2]
                    t = ops.arange(len(targets[i]["labels"]), dtype=ms.int64)
                    # (dn_num, num_inst_i)
                    t = ms_np.tile(t.unsqueeze(0), (dn_num, 1))
                    tgt_idx = t.flatten()
                    # (dn_num, 1) + (dn_num, num_inst_i) -> (dn_num, num_inst_i)
                    output_idx = (ops.arange(dn_num, dtype=ms.int64) * single_padding).unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = Tensor([], dtype=ms.int64)

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num, **kwargs)
                )

            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = ops.Tensor(0.0)
            losses["loss_giou_dn"] = ops.Tensor(0.0)
            losses["loss_class_dn"] = ops.Tensor(0.0)

        for i in range(aux_num):  # num_decoder_layer - 1
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:  # loss is a str
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            output_known_lbs_bboxes_aux,
                            targets,
                            dn_idx,
                            num_boxes * dn_num,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = Tensor(0.0)
                l_dict["loss_giou_dn"] = Tensor(0.0)
                l_dict["loss_class_dn"] = Tensor(0.0)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses


class TwoStageCriterion(SetCriterion):
    """
    This class computes the loss for two-stage DETR.
    two stage loss will be computed, including:
    1) normal detr decoder loss
    2) encoder proposal loss
    """
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        self.two_stage_binary_cls = two_stage_binary_cls

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)  # [(ind_src, ind_tgt)], len(indices)=bs

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = Tensor([num_boxes], dtype=ms.float32,)
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
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = ops.zeros_like(bt["labels"])  # why not ones
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
