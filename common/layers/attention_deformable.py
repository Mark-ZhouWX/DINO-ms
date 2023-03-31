import math
import warnings
from typing import Optional, List

import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init
import mindspore.numpy as ms_np

from common.layers.multi_scale import get_prod_shape
from common.utils.work_around import split

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MultiScaleDeformableAttention(nn.Cell):
    """
    Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dim (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): The number of attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query
            in each head. Default: 4.
        img2col_steps (int): The step used in image_to_column. Defualt: 64.
        dropout (float): Dropout layer used in output. Default: 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, but got {} and {}".format(
                    embed_dim, num_heads
                )
            )
        head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(keep_prob=1 - dropout)

        if not _is_power_of_2(head_dim):
            warnings.warn(
                """
                You'd better set d_model in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """
            )

        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Dense(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Dense(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Dense(embed_dim, embed_dim)
        self.output_proj = nn.Dense(embed_dim, embed_dim)

        self.init_weights()

    def init_weights(self):
        """
        Default initialization for Parameters of Module.
        """
        pshape, dtype = self.sampling_offsets.weight.shape, self.sampling_offsets.weight.dtype
        self.sampling_offsets.weight.set_data(init.initializer('zeros', pshape, dtype))
        thetas = ops.arange(self.num_heads).astype(ms.float32) * (2.0 * math.pi / self.num_heads)  # (num_head,)
        grid_init = ops.stack([ops.cos(thetas), ops.sin(thetas)], -1)  # (num_head, 2)
        grid_init = ms_np.tile(
            # (num_head, 1) -> (8, 1) -> (num_head, 2)
            (grid_init / grid_init.abs().max(-1, keepdims=True)[0])
            .view(self.num_heads, 1, 1, 2)
            , (1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias = ms.Parameter(grid_init.view(-1))
        self.sampling_offsets.bias = ops.stop_gradient(self.sampling_offsets.bias)

        pshape, dtype = self.attention_weights.weight.shape, self.attention_weights.weight.dtype
        self.attention_weights.weight.set_data(init.initializer('zeros', pshape, dtype))

        pshape, dtype = self.attention_weights.bias.shape, self.attention_weights.bias.dtype
        self.attention_weights.bias.set_data(init.initializer('zeros', pshape, dtype))

        pshape, dtype = self.value_proj.weight.shape, self.value_proj.weight.dtype
        self.value_proj.weight.set_data(init.initializer(init.XavierUniform(), pshape, dtype))

        pshape, dtype = self.value_proj.bias.shape, self.value_proj.bias.dtype
        self.value_proj.bias.set_data(init.initializer('zeros', pshape, dtype))

        pshape, dtype = self.output_proj.weight.shape, self.output_proj.weight.dtype
        self.output_proj.weight.set_data(init.initializer(init.XavierUniform(), pshape, dtype))

        pshape, dtype = self.output_proj.bias.shape, self.output_proj.bias.dtype
        self.output_proj.bias.set_data(init.initializer('zeros', pshape, dtype))

    @ms.ms_function
    def construct(
        self,
        query: Tensor,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[List] = None,
    ) -> Tensor:

        """
        Defines the computation to be performed.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default: None. If None, `query` will be
                used.
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (List[List]): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        # assert int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()) == num_value
        assert sum(get_prod_shape(spatial_shapes)) == num_value

        # assert ops.equal(ops.prod(spatial_shapes.astype(ms.float32), axis=1).sum(), num_value)

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)  # (bs, sum(hw), num_head, head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        # softmax to aggregate features of different level
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = ops.softmax(attention_weights, -1)  # (bs, sum(hw), num_head, num_level*num_point)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            spatial_shapes_tensor = Tensor(spatial_shapes, ms.int32)
            # (num_level, 2), 2: wh
            offset_normalizer = ops.stack([spatial_shapes_tensor[..., 1], spatial_shapes_tensor[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )  # (bs, sum(hw), num_heads, num_levels, num_points, 2)
        elif reference_points.shape[-1] == 4:
            # modulate xy offset by hw
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return self.dropout(output) + identity


def multi_scale_deformable_attn(
        value: Tensor,  # (bs, sum(hw), num_head, head_embed_dims)  head_embed_dims=embed_dim//num_head
        value_spatial_shapes: List,  # (num_level, 2)
        sampling_locations: Tensor,  # (bs, num_query, num_head, num_level, num_points, 2), normalized
        attention_weights: Tensor,
    ) -> Tensor:

    bs, _, num_heads, head_embed_dims = value.shape  # embed_dim is the one for head
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # indices_or_sections = ops.cumsum(value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1], axis=0)[:-1]
    split_sections = get_prod_shape(value_spatial_shapes)
    # split_sections = ops.prod(value_spatial_shapes.astype(ms.float32), axis=1).astype(ms.int32)
    # value_list = ops.split(value, split_sections, axis=1)
    value_list = split(value, split_sections, axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    # value_spatial_shapes_list = value_spatial_shapes.astype(ms.float32).asnumpy().tolist()
    value_spatial_shapes_list = value_spatial_shapes
    for level, (H_, W_) in enumerate(value_spatial_shapes_list):
        # H_, W_ = int(H_), int(W_)
        # bs, H_*W_, num_heads, head_embed_dims ->
        # bs, H_*W_, num_heads*head_embed_dims ->
        # bs, num_heads*head_embed_dims, H_*W_ ->
        # bs*num_heads, head_embed_dims, H_, W_
        value_l_ = (
            value_list[level].reshape(bs, H_ * W_, -1).transpose((0, 2, 1)).reshape(bs * num_heads, head_embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose((0, 2, 1, 3, 4)).reshape(
            bs * num_heads, num_queries, num_points, 2)
        # bs*num_heads, head_embed_dims, num_queries, num_points
        sampling_value_l_ = ops.grid_sample(
            value_l_, sampling_grid_l_, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose((0, 2, 1, 3, 4)).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )

    # (bs*num_heads, head_embed_dims, num_queries, num_levels, num_points) ->
    # (bs*num_heads, head_embed_dims, num_queries, num_levels*num_points) ->
    # (bs*num_heads, head_embed_dims, num_queries, num_levels*num_points) ->
    # (bs*num_heads, head_embed_dims, num_queries) -> [aggregate among level and pts axis]
    # (bs, num_heads*head_embed_dims, num_queries)
    output = (
        (ops.stack(sampling_value_list, axis=-2).reshape(bs * num_heads, head_embed_dims, num_queries, -1) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * head_embed_dims, num_queries)
    )
    # (bs, num_queries, embed_dims)  embed_dims = num_heads*head_embed_dims
    return output.transpose((0, 2, 1))
