from mindspore import nn, ops

_grad_scale = ops.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class TrainOneStepWithGradClipLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class with gradient clip.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self, network, optimizer, scale_sense=1, grad_clip=False, clip_value=0.1, ema=None):
        if isinstance(scale_sense, (int, float)):
            scale_sense = nn.FixedLossScaleUpdateCell(scale_sense)
        super(TrainOneStepWithGradClipLossScaleCell, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip
        self.grad_clip_value = clip_value
        self.ema = ema
        ema_not_none = ema is not None
        print(f'ema_not_none {ema_not_none}')

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)

        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)

        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.grad_clip_value)

        grads = self.grad_reducer(grads)

        cond = self.get_overflow_status(status, grads)

        overflow = self.process_loss_scale(cond)

        if not overflow:
            self.optimizer(grads)
            if self.ema is not None:
                self.ema.ema_update()

        return loss, cond, scaling_sens
