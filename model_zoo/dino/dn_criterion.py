import mindspore as ms
import mindspore.numpy as ms_np
from mindspore import Tensor, ops

from common.detr.criterion.set_criterion import SetCriterion
from common.utils.misc import replace_invalid


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

        # Compute all the requested losses
        base_loss_names = ['loss_class', 'loss_bbox', 'loss_giou']
        loss_names = []
        loss_values = []

        # get 3 basic loss, label, bbox and giou
        loss_last_decoder = self.get_loss(outputs_last_encoder, self.get_matched_target(outputs_last_encoder, targets))
        loss_names.extend(base_loss_names)
        loss_values.extend(loss_last_decoder)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if outputs_auxiliary is not None:
            aux_len = len(outputs_auxiliary[0])
            for i in range(aux_len):
                aux_out = (outputs_auxiliary[0][i], outputs_auxiliary[1][i])
                loss_aux = self.get_loss(aux_out, self.get_matched_target(outputs_last_encoder, targets))
                loss_names.extend([k + f"_{i}" for k in base_loss_names])
                loss_values.extend(loss_aux)

        # for two stage
        if outputs_encoder is not None:
            if self.two_stage_binary_cls:
                # reset target label, 0 means object, 1-79 no object
                for bt in targets:
                    bt["labels"] = ops.zeros_like(bt["labels"])
            loss_enc = self.get_loss(outputs_encoder, self.get_matched_target(outputs_last_encoder, targets))
            loss_names.extend([k + "_enc" for k in base_loss_names])
            loss_values.extend(loss_enc)

        losses = {k: v for k, v in zip(loss_names, loss_values)}
        return losses

class DINOCriterion(TwoStageCriterion):
    """
    This class computes the loss for DINO.
    Add dn loss to TwoStageCriterion, two type of loss will be computed:
    1) two stage loss, including normal detr decoder loss and encoder proposal loss
    2) dn and its auxiliary loss
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
        num_dn: float = 100,
    ):
        super(DINOCriterion, self).__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma, two_stage_binary_cls)
        self.num_dn = num_dn

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
        loss_dict = self.compute_two_stage_loss(outputs[:3], targets[:3])

        # Compute all the requested losses
        dn_loss_dict = self.compute_dn_loss(outputs[3:5], targets)
        loss_dict.update(dn_loss_dict)

        weighted_loss = self.compute_weighted_loss(loss_dict)

        return weighted_loss

    def compute_dn_loss(self, outputs, targets):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            aux_num: aux loss number
            targets (List[Dict]): list with length of batch_size,contains instances of one batch
            num_boxes: total number of boxes within a batch
        """
        last_decoder, auxiliary = outputs[0], outputs[1]

        # Compute all the requested losses
        base_loss_names = ['loss_class_dn', 'loss_bbox_dn', 'loss_giou_dn']
        loss_names = []
        loss_values = []

        # get 3 basic loss, label, bbox and giou
        if last_decoder is not None:
            loss_last_decoder = self.get_loss(last_decoder, self.get_cdn_targets(last_decoder, targets))
            loss_names.extend(base_loss_names)
            loss_values.extend(loss_last_decoder)

        # auxiliary losses
        if auxiliary is not None:
            aux_len = len(auxiliary[0])
            for i in range(aux_len):
                aux_out = (auxiliary[0][i], auxiliary[1][i])
                loss_aux = self.get_loss(aux_out, self.get_cdn_targets(aux_out, targets))
                loss_names.extend([k + f"_{i}" for k in base_loss_names])
                loss_values.extend(loss_aux)

        losses = {k: v for k, v in zip(loss_names, loss_values)}
        return losses

    @ms.ms_function
    def get_cdn_targets(self, outputs, targets):
        """
        get contrastive de-noising targets, including classes amd boxes and their valid masks
        Parameters:
            outputs (Tuple[Tensor]): pred_logits and pred_boxes
            targets (Tuple[Tensor]): raw target classes and boxes
        Returns:
            cdn_labels (Tensor[bs, num_cdn]): cdn target classes
            cdn_boxes (Tensor[bs, num_cdn, 4]): cdn target boxes
            cdn_valids (Tensor[bs, num_cdn]): valid mask of target matches
        """
        src_logits = outputs[0]
        bs, num_query, _ = src_logits.shape
        num_cdn = self.num_dn * 2
        assert num_cdn == num_query

        tgt_labels, tgt_boxes, tgt_valids = targets[:3]
        bs, num_pad_box = tgt_labels.shape
        tgt_labels = replace_invalid(tgt_labels, tgt_valids, self.num_classes)
        num_valid_box = ops.reduce_sum(tgt_valids.astype(ms.float32), 1).astype(ms.int32)  # (bs)
        dn_valids = targets[3]  # (bs, num_dn)
        assert self.num_dn == dn_valids.shape[1]

        src_ind = ms_np.tile(ops.expand_dims(ms_np.arange(self.num_dn), 0), (bs, 1))  # (bs, num_dn)
        src_ind = replace_invalid(src_ind, dn_valids, num_cdn - 1)  # 012 345 789 199

        tgt_ind = ops.expand_dims(ms_np.arange(self.num_dn), 0) % num_valid_box.expand_dims(1)  # (bs, num_dn)
        tgt_ind = replace_invalid(tgt_ind, dn_valids, num_pad_box - 1)  # 012 012 012 99

        cdn_labels = ms_np.full((bs, num_cdn), self.num_classes, dtype=ms.float32)
        sorted_dl = ops.gather_elements(tgt_labels, dim=1, index=tgt_ind).astype(ms.float32)  # (bs, num_dn)
        cdn_labels = ops.tensor_scatter_elements(cdn_labels, indices=src_ind, updates=sorted_dl, axis=1).astype(ms.int32)

        cdn_boxes = ms_np.full((bs, num_cdn, 4), 0.0, dtype=ms.float32)
        sorted_db = ops.gather_elements(tgt_boxes, dim=1, index=ms_np.tile(ops.expand_dims(tgt_ind, -1), (1, 1, 4)))
        cdn_boxes = ops.tensor_scatter_elements(cdn_boxes, indices=ms_np.tile(ops.expand_dims(src_ind, -1), (1,1, 4)),
                                                updates=sorted_db, axis=1)

        # valid masks
        cdn_valids = ops.concat([dn_valids, ops.zeros_like(dn_valids)], axis=1)  # (bs, num_cdn)

        return cdn_labels, cdn_boxes, cdn_valids
