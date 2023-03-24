import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist as np_cdist, cdist

from common.dataset.transform import box_cxcywh_to_xyxy


def sigmoid(x):
    return 1.0/ (1 + np.exp(-x))


def generalized_box_iou(boxes1, boxes2):
    """
    boxes1 shape : shape (n, 4)
    boxes2 shape : shape (k, 4)
    gious: shape (n, k)
    """
    IOU = []
    GIOU = []
    num = (boxes1[:, 0]).size
    x1 = boxes1[:, 0]
    y1 = boxes1[:, 1]
    x2 = boxes1[:, 2]
    y2 = boxes1[:, 3]

    xx1 = boxes2[:, 0]
    yy1 = boxes2[:, 1]
    xx2 = boxes2[:, 2]
    yy2 = boxes2[:, 3]

    area1 = (x2 - x1) * (y2 - y1)  # 求取框的面积
    area2 = (xx2 - xx1) * (yy2 - yy1)
    for i in range(num):
        inter_max_x = np.minimum(x2[i], xx2[:])  # 求取重合的坐标及面积
        inter_max_y = np.minimum(y2[i], yy2[:])
        inter_min_x = np.maximum(x1[i], xx1[:])
        inter_min_y = np.maximum(y1[i], yy1[:])
        inter_w = np.maximum(0, inter_max_x - inter_min_x)
        inter_h = np.maximum(0, inter_max_y - inter_min_y)

        inter_areas = inter_w * inter_h

        out_max_x = np.maximum(x2[i], xx2[:])  # 求取包裹两个框的集合C的坐标及面积
        out_max_y = np.maximum(y2[i], yy2[:])
        out_min_x = np.minimum(x1[i], xx1[:])
        out_min_y = np.minimum(y1[i], yy1[:])
        out_w = np.maximum(0, out_max_x - out_min_x)
        out_h = np.maximum(0, out_max_y - out_min_y)

        outer_areas = out_w * out_h
        union = area1[i] + area2[:] - inter_areas  # 两框的总面积   利用广播机制
        ious = inter_areas / union
        gious = ious - (outer_areas - union) / outer_areas  # IOU - ((C\union）/C)
        IOU.append(ious)
        GIOU.append(gious)
    return np.stack(GIOU, axis=0)


class HungarianMatcherNumpy(nn.Cell):
    """HungarianMatcher which computes an assignment between targets and predictions.
    """

    def __init__(
            self,
            cost_class: float = 1,
            cost_bbox: float = 1,
            cost_giou: float = 1,
            cost_class_type: str = "focal_loss_cost",
            alpha: float = 0.25,
            gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class_type = cost_class_type
        self.alpha = alpha
        self.gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        assert cost_class_type in {
            "ce_cost",
            "focal_loss_cost",
        }, "only support ce loss or focal loss for computing class cost"

    def construct(self, outputs, targets):
        """Forward function for `HungarianMatcher` which performs the matching.
            For each batch element, it holds: `len(index_i) = len(index                                                                                                                                                                                                                                                                                                                                                                                               _j) = min(num_queries, num_target_boxes)`
        """
        pred_logits, pred_boxes = outputs[0].asnumpy(), outputs[1].asnumpy()
        # (bs, num_box)   (bs, num_box, 4)   (bs, num_box)
        bs_tgt_labels, bs_tgt_bboxes, bs_tgt_valids = targets[0].asnumpy(), targets[1].asnumpy(), targets[2].asnumpy()

        bs, num_queries, num_class = pred_logits.shape
        num_pad_box = bs_tgt_labels.shape[1]

        # replace -1 with num_class
        bs_tgt_labels *= bs_tgt_valids.astype(np.int32)
        bs_tgt_labels += np.logical_not(bs_tgt_valids).astype(np.int32) * (num_class-1)  # replace unvalid with num_class-1

        # Flatten batch to compute the cost matrices in a batch
        if self.cost_class_type == "ce_cost":
            out_prob = (
                ops.reshape(pred_logits, (bs * num_queries, -1)).softmax(-1)
            )  # [bs * num_queries, num_classes]
        elif self.cost_class_type == "focal_loss_cost":
            out_prob = sigmoid(np.reshape(pred_logits, (bs * num_queries, -1))) # [bs * num_queries, num_classes]
        else:
            raise NotImplementedError(f'support loss_type [ce_cost], [ce_cost], but got {self.cost_class_type}')
        out_bbox = np.reshape(pred_boxes, (bs * num_queries, -1))  # [batch_size * num_queries, 4]

        # Flatten batch
        tgt_ids = bs_tgt_labels.reshape(-1)  # (bs*num_box,)
        tgt_bbox = bs_tgt_bboxes.reshape(bs * num_pad_box, -1)  # (bs*num_box, 4)
        # tgt_valid = bs_tgt_valids.reshape(-1)  # (bs*num_box,)

        # Compute the classification cost.
        if self.cost_class_type == "ce_cost":
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]
        elif self.cost_class_type == "focal_loss_cost":
            alpha = self.alpha
            gamma = self.gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-np.log((1 - out_prob + 1e-8)))
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-np.log((out_prob + 1e-8)))
            # cost_class = ops.gather(pos_cost_class - neg_cost_class, tgt_ids, axis=1)
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            raise NotImplementedError(f'support only ce_cost and focal_loss_cost, '
                                      f'but got class_type {self.cost_class_type}')

        # Compute the L1 cost between boxes
        cost_bbox = cdist(out_bbox, tgt_bbox, metric='minkowski', p=1)  # (bs*num_queries, bs*num_box)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix, (bs*num_queries, bs*num_box)
        weighted_cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        # static version of weight matrix
        # (bs, bs, num_queries, num_box)
        weighted_cost_matrix = weighted_cost_matrix.reshape(bs, num_queries, bs, num_pad_box).transpose(2, 0, 1, 3)

        # two layer of index, the batch_i+batch_i is kept while batch_i+batch_other discarded
        weighted_cost_matrix_np = weighted_cost_matrix
        bs_tgt_valids_np = bs_tgt_valids

        assert num_pad_box <= num_queries, "query number should be bigger that gt box number"
        src_index = np.ones(shape=(bs, num_pad_box)) * -1
        tgt_index = np.ones(shape=(bs, num_pad_box)) * -1

        for i in range(bs):
            # ops.masked_select(weighted_cost_matrix[i, i], bs_tgt_valids[i][None, :])
            valid_wcm = weighted_cost_matrix_np[i, i][:, bs_tgt_valids_np[i]]
            row_, col_ = linear_sum_assignment(valid_wcm)
            src_index[i, :len(row_)] = row_
            tgt_index[i, :len(col_)] = col_

        return ops.stop_gradient(Tensor(src_index, ms.int32)), ops.stop_gradient(Tensor(tgt_index, ms.int32))
