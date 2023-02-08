from functools import partial

from mindspore import nn


class ConvNormAct(nn.Cell):
    """
    Utility module that stacks one convolution 2D layer,
    a normalization layer and an activation function.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): Size of the convolving kernel. Default: 1.
        stride (int): Stride of convolution. Default: 1.
        padding (int): Padding added to all four sides of the input. Default: 0.
        dilation (int): Spacing between kernel elements. Default: 1.
        group (int): Number of blocked connections from input channels
            to output channels. Default: 1.
        bias (bool): if True, adds a learnable bias to the output. Default: True.
        norm_layer (nn.Module): Normalization layer used in `ConvNormAct`. Default: None.
        activation (nn.Module): Activation layer used in `ConvNormAct`. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        group: int = 1,
        bias: bool = True,
        norm_layer: nn.Cell = None,
        activation: nn.Cell = None,
        **kwargs,
    ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            group=group,
            has_bias=bias,
            **kwargs,
        )
        self.norm = norm_layer
        self.activation = activation

    def construct(self, x):
        """Forward function for `ConvNormAct`"""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


ConvNorm = partial(ConvNormAct, activation=None)
