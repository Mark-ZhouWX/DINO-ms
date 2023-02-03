from mindspore import Tensor, ops


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    Args:
        x (Tensor) : tensor within range 0,1
        eps (float) :
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return ops.log(x1 / x2)
