import mindspore as ms
from mindspore import Tensor, ops


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    Args:
        x (Tensor) : tensor within range 0,1
        eps (float) :
    """
    x = ops.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    x1 = ops.clip_by_value(x, clip_value_min=Tensor(eps))
    x2 = ops.clip_by_value(1 - x, clip_value_min=Tensor(eps))
    return ops.log(x1 / x2)


def replace_invalid(inputs, v_mask, value, dtype=ms.int32):
    """
    replace value of invalid index to the given value
    Args:
        inputs (Tensor)： inputs tensor
        v_mask (Tensor): mask that indicates valid index
        value (int, float): value to replace
        dtype (ms.number): output date type
    """
    res = inputs * v_mask.astype(dtype)
    res += ops.logical_not(v_mask).astype(dtype) * value  # replace invalid with given value
    return res.astype(dtype)