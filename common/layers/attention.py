import warnings
from typing import Optional

import mindspore as ms
import mindspore.numpy as ms_np
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init


class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.
    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ops = Attention(768, 12)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct."""
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out


class MultiheadAttention(nn.Cell):
    """

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads, but got {} and {}".format(embed_dim, num_heads))

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.out_drop = nn.Dropout(keep_prob=1 - proj_drop)

        # proj qkv, separately reference : torch.MultiHeadAttention
        # self.in_proj_weight = ms.Parameter(init.initializer(init.XavierUniform,
        #                                                     dtype=ms.float32, shape=(embed_dim * 3, embed_dim)))
        # self.in_proj_bias = ms.Parameter(init.initializer(init.Constant,
        #                                                   dtype=ms.float32, shape=(embed_dim * 3)))
        self.in_projs = nn.CellList([nn.Dense(embed_dim, embed_dim),
                                    nn.Dense(embed_dim, embed_dim), nn.Dense(embed_dim, embed_dim)])
        self.out_proj = nn.Dense(embed_dim, embed_dim)  # proj z

        self.head_dim = embed_dim // num_heads

        self.softmax_scale = Tensor(self.head_dim ** -0.5)

    def init_weights(self):
        # TODO init weight for MSA
        pass

    def construct(
            self,
            query: Tensor,
            key: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            identity: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            key_pos: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            **kwargs,
    ) -> Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        # prepare
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is" f"missing in {self.__class__.__name__}.")
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # embed_dim of query, key and value must be the same as self.embed_dim
        bs, num_query, _ = query.shape
        _, num_key, _ = key.shape
        _, num_value, _ = value.shape
        assert num_value == num_key

        query, key, value = [self.in_projs[i](t) for i, t in enumerate([query, key, value])]
        query = query.reshape(bs, num_query, self.num_heads, self.head_dim)
        key = key.reshape(bs, num_key, self.num_heads, self.head_dim)
        value = value.reshape(bs, num_key, self.num_heads, self.head_dim)
        # query = ops.reshape(query, (bs, self.num_heads, num_query, self.head_dim))
        # key = ops.reshape(key, (bs, self.num_heads, num_key, self.head_dim))
        # value = ops.reshape(value, (bs, self.num_heads, num_key, self.head_dim))

        # (bs, num_query, num_head, head_dim) -> (bs, num_head, num_query, head_dim)
        query, key, value = ops.transpose(query, (0, 2, 1, 3)), \
            ops.transpose(key, (0, 2, 1, 3)), ops.transpose(value, (0, 2, 1, 3))

        attn_output_weights = ops.BatchMatMul(transpose_b=True)(query, key)  # (bs, num_head, num_query, num_key)
        attn_output_weights = ops.mul(attn_output_weights, self.softmax_scale)

        if attn_mask is not None:
            if attn_mask.dtype != ms.bool_:
                raise ValueError(f'attention mask type should be bool, but got {attn_mask.dtype} instead')
            if attn_mask.ndim == 3: # (bs, num_query, num_key) -> (bs, 1, num_query, num_key)
                attn_mask = attn_mask.expand_dims(1)
            attn_output_weights = ops.masked_fill(attn_output_weights, attn_mask, float("-inf"))
        if key_padding_mask is not None:
            if key_padding_mask.dtype != ms.bool_:
                raise ValueError(f'key padding mask type should be bool, but got {attn_mask.dtype} instead')
            # (bs, 1, 1, num_key)  -> (bs, num_head, num_query, num_key)
            attn_output_weights = ops.masked_fill(attn_output_weights,
                                                  key_padding_mask[:, None, None, :], float("-inf"))

        # (bs, num_head, num_query, num_key) -> (bs, num_head, num_query, num_key)
        attn = ops.softmax(attn_output_weights, -1)
        attn = self.attn_drop(attn)

        out = ops.BatchMatMul()(attn, value)  # (bs, num_head, num_query, head_dim)
        out = ops.transpose(out, (0, 2, 1, 3))  # (bs, num_query, num_head, head_dim)
        out = ops.reshape(out, (bs, num_query, self.embed_dim))
        out = self.out_proj(out)
        out = self.out_drop(out)

        return identity + out
