import mindspore as ms
import mindspore.numpy as ms_np
from mindspore import Tensor, ops


def split(value, indices, axis):
    assert isinstance(indices, (int, list))
    if isinstance(indices, int):
        total = value.shape[axis]
        count = total // indices
        indices = [indices for _ in range(count)]
    split_ends = ops.cumsum(Tensor(indices, ms.int32), 0)
    new_indices = indices.copy()
    new_indices.pop()
    new_indices.insert(0, 0)
    split_starts = ops.cumsum(Tensor(new_indices, ms.int32), 0)
    outs = []
    for s, e in zip(split_starts, split_ends):
        out = ops.gather(value, ms_np.arange(s, e, Tensor(1, s.dtype), dtype=s.dtype), axis=axis)
        outs.append(out)
    return outs


def cdist(x, y, p=1.0):
    print(x.shape, y.shape)
    assert p == 1.0
    assert len(x.shape) == 2

    if p == 1.0:
        abs_diff = ops.abs(x[:, None] - y[None, :])
        p_dist_mat = ops.reduce_sum(abs_diff, axis=-1)
    else:
        raise NotImplementedError
    
    return p_dist_mat
