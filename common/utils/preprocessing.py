from typing import List

import mindspore as ms
from mindspore import Tensor, ops
import mindspore.numpy as ms_np


def pad_as_batch(image_list: List[Tensor], pad_value=0.0, size_divisibility=0):
    """
    pad image list as a Tensor with shape (bs, c, h, w), with hw the max value of that of the image
    """
    assert isinstance(image_list, (tuple, list))
    for t in image_list:
        assert isinstance(t, Tensor), type(t)
        assert t.shape[:-2] == image_list[0].shape[:-2], t.shape
    bs = len(image_list)
    c, h, w = image_list[0].shape
    image_sizes = [(im.shape[-2], im.shape[-1]) for im in image_list]
    image_sizes_tensor = [Tensor(x) for x in image_sizes]
    max_size = ops.max(ops.stack(image_sizes_tensor).astype(ms.float32), 0)[1]  # (2,)

    if size_divisibility > 1:
        stride = size_divisibility
        max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

    batch_shape = [bs, c, int(max_size[0]), int(max_size[1])]
    batched_imgs = ms_np.full(batch_shape, pad_value)
    for i_bs, img in enumerate(image_list):
        batched_imgs[i_bs, ..., :image_sizes[i_bs][0], :image_sizes[i_bs][1]] = img

    return batched_imgs, image_sizes
