import mindspore as ms
import mindspore.numpy as ms_np
from mindspore import Tensor, ops

from common.layers.multi_scale import get_list_cum_sum


def split(value, indices, axis):
    assert isinstance(indices, (int, list))

    if isinstance(indices, int):
        total = value.shape[axis]
        count = total // indices
        indices = [indices for _ in range(count)]

    split_starts, split_ends= get_list_cum_sum(indices, 2)
    outs = []
    count = len(indices)
    for i in range(count):
        s, e = split_starts[i], split_ends[i]
        out = ops.gather(value, ms_np.arange(s, e), axis=axis)
        outs.append(out)
    return outs


def cdist(x, y, p=1.0):
    assert p == 1.0
    assert len(x.shape) == 2

    if p == 1.0:
        abs_diff = ops.abs(x[:, None] - y[None, :])
        p_dist_mat = ops.reduce_sum(abs_diff, axis=-1)
    else:
        raise NotImplementedError
    
    return p_dist_mat
