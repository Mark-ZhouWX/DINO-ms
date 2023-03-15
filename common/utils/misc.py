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
