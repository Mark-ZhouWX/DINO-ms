import mindspore as ms
import mindspore.numpy as ms_np
from mindspore import Tensor, ops

from common.detr.criterion.set_criterion import SetCriterion


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
        use_np_mather: bool = False,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma, use_np_mather
        )
        self.two_stage_binary_cls = two_stage_binary_cls

    def construct(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs (Tuple[Tuple[Tensor]]): predictions of last decoder, auxiliary, encoder, each prediction contains
                                            a tuple with label and bbox.
             targets (Tuple[Tensor]): target tuple contains gt label, box and valid_mask

        Returns:
             loss (tuple(Tensor): two_stage loss with size 3, (last, aux, encoder), each tensor contains three type of loss, (bbox, giou, class)
        """
        self.compute_two_stage_loss(outputs, targets)


    def compute_two_stage_loss(self, outputs, targets):
        outputs_last_encoder = outputs[0]
        outputs_auxiliary =  outputs[1]
        outputs_encoder = outputs[2]
        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_last_encoder, targets)  # [(ind_src, ind_tgt)], len(indices)=bs

        # Compute all the requested losses
        losses = {}

        # get 3 basic loss, label, bbox and giou
        loss_last_decoder = self.get_loss(outputs_last_encoder, self.get_matched_target(outputs_last_encoder, targets))
        losses.update(loss_last_decoder)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if outputs_auxiliary is not None:
            aux_len = len(outputs_auxiliary[0])
            for i in range(aux_len):
                aux_out = (outputs_auxiliary[0][i], outputs_auxiliary[1][i])
                loss_aux = self.get_loss(aux_out, self.get_matched_target(aux_out, targets))
                l_dict = {k + f"_{i}": v for k, v in loss_aux.items()}
                losses.update(l_dict)

        # for two stage
        if outputs_encoder is not None:
            if self.two_stage_binary_cls:
                # reset target label, 0 means object, 1-79 no object
                for bt in targets:
                    bt["labels"] = ops.zeros_like(bt["labels"])
            loss_enc = self.get_loss(outputs_encoder, self.get_matched_target(outputs_encoder, targets))
            l_dict = {k + "_enc": v for k, v in loss_enc.items()}
            losses.update(l_dict)

        return losses

class DINOCriterion(TwoStageCriterion):
    """
    This class computes the loss for DINO.
    Add dn loss to TwoStageCriterion, two type of loss will be computed:
    1) two stage loss, including normal detr decoder loss and encoder proposal loss
    2) dn and its auxiliary loss
    """

    def construct(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             dn_metas: de-noising information, including dn predicts and dn_number, etc.
        """
        # two_stage loss (tuple(Tensor)) with size 3 -> last, aux, encoder
        # each tensor contains three type of loss -> bbox, giou, class
        loss_dict = self.compute_two_stage_loss(outputs, targets)

        # Compute all the requested losses
        outputs_auxiliary = outputs[1]
        aux_num = len(outputs_auxiliary[0]) if outputs_auxiliary is not None else 0
        dn_loss_dict = self.compute_dn_loss(dn_metas, targets, aux_num)
        loss_dict.update(dn_loss_dict)

        weighted_loss = self.compute_weighted_loss(loss_dict)

        return weighted_loss

    def compute_dn_loss(self, dn_metas, targets, aux_num):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            aux_num: aux loss number
            targets (List[Dict]): list with length of batch_size,contains instances of one batch
            num_boxes: total number of boxes within a batch
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            for tgt in targets:
                # positive box assigned index, negative use default no object class, and no box regression supervision
                if len(tgt["labels"]) > 0:
                    # eg [0, 1, 2]
                    t = ops.arange(len(tgt["labels"]), dtype=ms.int64)
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
