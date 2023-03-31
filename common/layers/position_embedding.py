import math

import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore import numpy as ms_np
from common.utils.work_around import split


class PositionEmbeddingSine(nn.Cell):
    """Sinusoidal position embedding used in DETR model.

    Please see `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for more details.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default: 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default: 2*pi.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default: 1e-6.
        offset (float): An offset added to embed when doing normalization.
        normalize (bool, optional): Whether to normalize the position embedding.
            Default: False.
    """

    def __init__(
            self,
            num_pos_feats: int = 64,
            temperature: int = 10000,
            scale: float = 2 * math.pi,
            eps: float = 1e-6,
            offset: float = 0.0,
            normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def construct(self, mask: Tensor) -> Tensor:
        """Forward function for `PositionEmbeddingSine`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, h, w)`.

        Returns:
            torch.Tensor: Returned position embedding with shape `(bs, num_pos_feats * 2, h, w)`
        """
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=ms.float32)
        x_embed = not_mask.cumsum(2, dtype=ms.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = ops.arange(self.num_pos_feats).astype(ms.float32)
        dim_t = self.temperature ** (
                2 * ops.floor_div(dim_t, 2) / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # use view as mmdet instead of flatten for dynamically exporting to ONNX
        B, H, W = mask.shape
        pos_x = ops.stack((ops.sin(pos_x[:, :, :, 0::2]), ops.cos(pos_x[:, :, :, 1::2])), axis=4).view(
            B, H, W, -1
        )
        pos_y = ops.stack((ops.sin(pos_y[:, :, :, 0::2]), ops.cos(pos_y[:, :, :, 1::2])), axis=4).view(
            B, H, W, -1
        )
        pos = ops.concat((pos_y, pos_x), axis=3).transpose(0, 3, 1, 2)
        return pos

@ms.ms_function
def get_sine_pos_embed(pos_tensor: Tensor, num_pos_feats: int = 128, temperature: int = 10000, exchange_xy: bool = True,
                       ) -> Tensor:
    """generate sine position embedding from a position tensor

    Args:
        pos_tensor (torch.Tensor): Shape as `(None, n)`.
        num_pos_feats (int): projected shape for each float in the tensor. Default: 128
        temperature (int): The temperature used for scaling
            the position embedding. Default: 10000.
        exchange_xy (bool, optional): exchange pos of the first two dimension. \
            For example, input tensor is `[x, y]`, the results will  # noqa
            be `[pos(y), pos(x)]`. Defaults: True.

    Returns:
        torch.Tensor: Returned position embedding  # noqa
        with shape `(None, n * num_pos_feats)`.
    """
    scale = 2 * math.pi
    dim_t = ms_np.arange(num_pos_feats).astype(ms.float32)
    dim_t = temperature ** (2 * ops.floor_div(dim_t, 2) / num_pos_feats)

    def sine_func(x: Tensor):
        sin_x = x * scale / dim_t
        sin_x = ops.stack((ops.sin(sin_x[:, :, 0::2]), ops.cos(sin_x[:, :, 1::2])), axis=3)
        sin_x = sin_x.reshape(sin_x.shape[0], sin_x.shape[1], -1)
        return sin_x

    pos_res = [sine_func(x) for x in split(pos_tensor, 1, axis=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = ops.concat(pos_res, axis=2)
    return pos_res
