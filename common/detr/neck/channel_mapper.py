import copy
from typing import Dict, List

import mindspore as ms
from mindspore import nn

from common.layers.conv import ConvNormAct


class ChannelMapper(nn.Cell):
    """Channel Mapper for reduce/increase channels of backbone features. Modified
    from `mmdet <https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/channel_mapper.py>`_.

    This is used to reduce/increase the channels of backbone featuresa, and usually won't change the feature size.

    Args:
        input_shape (Dict[str, ShapeSpec]): A dict which contains the backbone features meta infomation,
            e.g. ``input_shape = {"res5": ShapeSpec(channels=2048)}``.
        in_features (List[str]): A list contains the keys which maps the features output from the backbone,
            e.g. ``in_features = ["res"]``.
        out_channels (int): Number of output channels for each scale.
        kernel_size (int, optional): Size of the convolving kernel for each scale.
            Default: 3.
        stride (int, optional): Stride of convolution for each scale. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output of each scale.
            Default: True.
        group (int, optional): Number of blocked connections from input channels to
            output channels for each scale. Default: 1.
        dilation (int, optional): Spacing between kernel elements for each scale.
            Default: 1.
        norm_layer (nn.Module, optional): The norm layer used for each scale. Default: None.
        activation (nn.Module, optional): The activation layer used for each scale. Default: None.
        num_outs (int, optional): Number of output feature maps. There will be ``extra_convs`` when
            ``num_outs`` is larger than the length of ``in_features``. Default: None.
    """

    def __init__(
        self,
        input_channels: Dict[str, int],
        in_features: List[str],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        group: int = 1,
        dilation: int = 1,
        norm_layer: nn.Cell = None,
        activation: nn.Cell = None,
        num_outs: int = None,
        **kwargs,
    ):
        super(ChannelMapper, self).__init__()
        self.extra_convs = None

        in_channels_per_feature = [input_channels[f] for f in in_features]

        if num_outs is None:
            num_outs = len(input_channels)

        self.convs = nn.CellList()
        for in_channel in in_channels_per_feature:
            self.convs.append(
                ConvNormAct(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    group=group,
                    dilation=dilation,
                    norm_layer=copy.deepcopy(norm_layer),
                    activation=copy.deepcopy(activation),
                )
            )

        if num_outs > len(in_channels_per_feature):
            self.extra_convs = nn.CellList()
            for i in range(len(in_channels_per_feature), num_outs):
                if i == len(in_channels_per_feature):
                    in_channel = in_channels_per_feature[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvNormAct(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        pad_mode='pad',
                        bias=bias,
                        group=group,
                        dilation=dilation,
                        norm_layer=copy.deepcopy(norm_layer),
                        activation=copy.deepcopy(activation),
                    )
                )

        self.input_channels = input_channels
        self.in_features = in_features
        self.out_channels = out_channels

    @ms.ms_function
    def construct(self, inputs):
        """Forward function for ChannelMapper

        Args:
            inputs (Tuple[torch.Tensor]): The backbone feature maps.

        Return:
            tuple(torch.Tensor): A tuple of the processed features.
        """
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)
