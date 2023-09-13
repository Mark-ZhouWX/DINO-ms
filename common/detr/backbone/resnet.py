"""
MindSpore implementation of `ResNet`.
Refer to Deep Residual Learning for Image Recognition.
"""
import logging
import os
from functools import partial
from typing import Optional, Type, List, Union, Dict

from mindspore import nn, Tensor, load_checkpoint, load_param_into_net


__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x4d',
    'resnext101_64x4d',
    'resnext152_64x4d'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'conv1', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'resnet18': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18_224.ckpt'),
    'resnet34': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnet/resnet34_224.ckpt'),
    'resnet50': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnet/resnet50_224.ckpt'),
    'resnet101': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101_224.ckpt'),
    'resnet152': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnet/resnet152_224.ckpt'),
    'resnext50_32x4d': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnext/resnext50_32x4d_224.ckpt'),
    'resnext101_32x4d': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnext/resnext101_32x4d_224.ckpt'),
    'resnext101_64x4d': _cfg(url=''),
    'resnext152_64x4d': _cfg(url='https://download.mindspore.cn/toolkits/mindcv/resnext/resnext152_64x4d_224.ckpt')
}


def load_pretrained(model, default_cfg, path='./', num_classes=1000, in_channels=3, filter_fn=None):
    '''load pretrained model depending on cfgs of model'''
    if 'url' not in default_cfg or not default_cfg['url']:
        logging.warning('Pretrained model URL is invalid')
        return

    # download files
    os.makedirs(path, exist_ok=True)
    download.download(default_cfg['url'], path=path)

    param_dict = load_checkpoint(os.path.join(path, os.path.basename(default_cfg['url'])))

    load_param_into_net(model, param_dict)


class BasicBlock(nn.Cell):
    """define the basic block of resnet"""
    expansion: int = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, 'BasicBlock only supports groups=1'
        assert base_width == 64, 'BasicBlock only supports base_width=64'

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, pad_mode='pad')
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, pad_mode='pad')
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """
    Bottleneck here places the stride for downsampling at 3x3 convolution(self.conv2) as torchvision does,
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    """
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad', group=groups)
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


def get_norm_from_str(norm_str):
    if norm_str == 'FrozenBN':
        bn = partial(nn.BatchNorm2d, affine=False, use_batch_statistics=False)
    elif norm_str == 'BN':
        bn = nn.BatchNorm2d
    else:
        raise NotImplementedError(f'require norm_str [FrozenBN], [BN], got [{norm_str}] instead')

    return bn


class ResNet(nn.Cell):
    r"""ResNet model class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        block: block of resnet.
        layers: number of layers of each stage.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 64.
        norm: normalization layer in blocks. Default: None.
        return_layers: define which layers to return
    """

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 out_layers: Optional[List[str]] = None,
                 ) -> None:
        super().__init__()
        self.all_layer_names = ["res1", "res2", "res3", "res4", "res5"]
        if out_layers is None:
            out_layers = ["res5"]
        for lay_str in out_layers:
            assert lay_str in self.all_layer_names
        self.out_layer_names = out_layers

        if norm is None:
            norm = nn.BatchNorm2d
        if isinstance(norm, str):
            norm = get_norm_from_str(norm)

        self.norm: nn.Cell = norm  # add type hints to make pylint happy
        self.input_channels = 64
        self.groups = groups
        self.base_with = base_width

        self.conv1 = nn.Conv2d(in_channels, self.input_channels, kernel_size=7,
                               stride=2, pad_mode='pad', padding=3)
        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, pad_mode='pad')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.num_features = 512 * block.expansion

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    channels: int,
                    block_nums: int,
                    stride: int = 1
                    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                nn.Conv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
                self.norm(channels * block.expansion)
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm
                )
            )

        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        """Network forward feature extraction."""
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu(x)

        x = self.max_pool(out1)  # different from torch in padding

        out2 = self.layer1(x)

        out3 = self.layer2(out2)

        out4 = self.layer3(out3)

        out5 = self.layer4(out4)

        out_layers = dict()
        for lay_name, out_feat in zip(self.all_layer_names, [out1, out2, out3, out4, out5]):
            if lay_name in self.out_layer_names:
                out_layers.update({lay_name: out_feat})

        return out_layers

    def construct(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.forward_features(x)
        return x


def resnet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 18 layers ResNet model.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnet18']
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnet34(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 34 layers ResNet model.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnet34']
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNet model.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnet50']
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnet101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNet model.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnet101']
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnet152(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 152 layers ResNet model.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnet152']
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnext50_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNeXt model with 32 groups of GPConv.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnext50_32x4d']
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, base_width=4, num_classes=num_classes,
                   in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnext101_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 32 groups of GPConv.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnext101_32x4d']
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, base_width=4, num_classes=num_classes,
                   in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnext101_64x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 64 groups of GPConv.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnext101_64x4d']
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=64, base_width=4, num_classes=num_classes,
                   in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


def resnext152_64x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['resnext101_64x4d']
    model = ResNet(Bottleneck, [3, 8, 36, 3], groups=64, base_width=4, num_classes=num_classes,
                   in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
