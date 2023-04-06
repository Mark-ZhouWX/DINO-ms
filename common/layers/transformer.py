import copy
import warnings
from typing import List, Tuple

import mindspore as ms
from mindspore import nn, Tensor

from common.layers.attention import MultiheadAttention
from common.layers.attention_deformable import MultiScaleDeformableAttention


class BaseTransformerLayer(nn.Cell):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.

    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn (list[nn.Cell] | nn.Cell): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
        self,
        attn: List[nn.Cell],
        ffn: nn.Cell,
        norm: nn.Cell,
        operation_order: tuple = None,
        attn_type: tuple = None,
    ):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        assert len(attn) == num_attn, (
            f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
            f"is not consistent with the number of attention in "
            f"operation_order {operation_order}"
        )
        assert len(attn_type) == num_attn

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.CellList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1
        self.attn_type = attn_type
        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        if not isinstance(ffn, nn.CellList):
            self.ffns = nn.CellList()
            num_ffns = operation_order.count("ffn")
            for _ in range(num_ffns):
                self.ffns.append(copy.deepcopy(ffn))
        else:
            self.ffns = ffn

        # count norm nums
        if not isinstance(norm, nn.CellList):
            self.norms = nn.CellList()
            num_norms = operation_order.count("norm")
            for _ in range(num_norms):
                self.norms.append(copy.deepcopy(norm))
        else:
            self.norms = norm

    # @ms.ms_function
    def construct(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        key_pos: Tensor = None,
        attn_masks: List[Tensor] = None,
        query_key_padding_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        reference_points: Tensor = None,
        spatial_shapes: Tuple = None,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (Tensor): The position embedding for `query`.
                Default: None.
            key_pos (Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
            multi_scale_args (Tuple[Tensor]): tuple that contains two args of multi-scale attention,
             namely, reference_points, spatial_shapes, level_start_index
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                if self.attn_type[attn_index] == 'MultiheadAttention':
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],  # None in encoder, active in decoder
                        key_padding_mask=query_key_padding_mask,  # None in decoder, active in encoder
                    )
                elif self.attn_type[attn_index] == 'MultiScaleDeformableAttention':
                    query = self.attentions[attn_index](
                        query,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_padding_mask=query_key_padding_mask,  # None in decoder, active in encoder
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                    )
                else:
                    raise NotImplementedError(f'not supported self-attetion type [{type(self.attentions[attn_index])}]')

                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                if self.attn_type[attn_index] == 'MultiheadAttention':
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                    )
                elif self.attn_type[attn_index] == 'MultiScaleDeformableAttention':
                    query = self.attentions[attn_index](
                        query,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_padding_mask=key_padding_mask,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                    )
                else:
                    raise NotImplementedError(f'not supported cross-attetion type [{type(self.attentions[attn_index])}]')

                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
