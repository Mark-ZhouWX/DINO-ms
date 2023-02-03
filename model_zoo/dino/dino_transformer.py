import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init

from common.layers.mlp import FFN, MLP
from common.layers.position_embedding import get_sine_pos_embed
from common.layers.transformer import TransformerLayerSequence, BaseTransformerLayer
from common.utils.misc import inverse_sigmoid


class DINOTransformer(nn.Cell):
    """Transformer module for DINO

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
        learnt_init_query (bool): whether to learn content query(static) or generate from two-stage proposal(dynamic)
    """

    def __init__(
            self,
            encoder=None,
            decoder=None,
            num_feature_levels=4,
            two_stage_num_proposals=900,
            learnt_init_query=True,
    ):
        super(DINOTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = ms.Parameter(Tensor(self.num_feature_levels, self.embed_dim))
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Dense(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def construct(
            self,
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
            **kwargs,
    ):
        """
        Args:
            multi_level_feats (List[Tensor[bs, embed_dim, h, w]]): list of multi level features from backbone(neck)
            multi_level_masks (List[Tensor[bs, h, w]]):list of masks of multi level features
            multi_level_pos_embeds (List[Tensor[bs, embed_dim, h, w]]):  list of pos_embeds multi level features
            query_embed (List[Tensor[bs, dn_number, embed_dim], Tensor[bs, dn_number, 4]]):
                len of list is 2, initial gt query for dn, including content_query and position query(reference point)
            attn_masks (List[Tensor]): attention map for dn
        """
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)  # multi-scale embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = ops.cat(feat_flatten, 1)
        mask_flatten = ops.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = ops.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = Tensor(spatial_shapes, dtype=ms.int64)
        level_start_index = ops.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # there may be slight difference of ratio values between different level due to of mask quantization
        valid_ratios = ops.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)  # (bs, num_level, 2)

        # reference_points for deformable-attn, range (H, W), un-normalized, flattened
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)  # (bs sum(hw), nl, 2)

        # (bs, sum(hw), c)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            # to mask image input padding area
            query_key_padding_mask=mask_flatten,
            # leave for deformable-attn
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # (bs, sum(hw), num_level, 2)
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        # (bs, sum(hw), c); (bs, sum(hw), 4) unsigmoided + valid
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # two-stage
        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided. (bs, sum(hw), 4)

        topk = self.two_stage_num_proposals

        # from mindspore import Tensor, ops
        # import mindspore.common.initializer as init
        # enc_outputs_class = Tensor(shape = (4, 8, 12), dtype=ms.float32, init=init.Uniform())
        # print(enc_outputs_class.shape)
        # print(enc_outputs_class.max(-1)[0])
        # topk = 3
        # TODO 此处应该把【0】去掉，待验证
        topk_proposals = ops.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # (bs, k) , k=num_query

        # extract region proposal boxes
        topk_coords_unact = ops.gather_elements(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
        )  # unsigmoided. (bs, k, 4)
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = ops.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = ops.gather_elements(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        if self.learnt_init_query:
            bs = multi_level_feats[0].shape[0]
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embed[0] is not None:
            target = ops.cat([query_embed[0], target], 1)

        # decoder
        inter_states, inter_references = self.decoder(
            query=target,  # (bs, sum(hw)+num_cdn, embed_dims) if dn training else None (bs, sum(hw), embed_dims)
            key=memory,  # bs, sum(hw), embed_dims
            value=memory,  # bs, sum(hw), embed_dims
            query_pos=None,
            # to mask input image padding area, active in cross_attention
            key_padding_mask=mask_flatten,  # bs, sum(hw)
            reference_points=reference_points,  # bs, sum(hw), 4
            spatial_shapes=spatial_shapes,  # (nlvl, 2)
            level_start_index=level_start_index,  # (nlvl)
            valid_ratios=valid_ratios,  # (bs, nlvl, 2)
            # to mask the information leakage between gt and matching queries, active in self-attention
            attn_masks=attn_masks,  # (bs, sum(hw)+num_cdn, sum(hw)+num_cdn) if dn training else None
            **kwargs,
        )

        inter_references_out = inter_references
        return (
            inter_states,
            init_reference_out,
            inter_references_out,
            target_unact,
            topk_coords_unact.sigmoid(),
        )

    def init_weights(self):
        for p in self.get_parameters():
            if p.dim() > 1:
                p.set_data(init.initializer(init.XavierUniform(), p.shape, p.dtype))
        for m in self.cells():
            # TODO to replace deformable attention with normal attention
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        self.level_embed.set_data(init.initializer(init.Uniform(), self.level_embeds.shape, self.level_embeds.dtype))

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """
        Args:
            memory (Tensor[bs, sum(hw), c]): flattened encoder memory
            memory_padding_mask (Tensor[bs, sum(hw)]): padding_mask of memory
            spatial_shapes (Tensor[num_layer, 2]): spatial shapes of multiscale layer
        Returns:
            Tensor[bs, sum(hw), c]: filtered memory
            Tensor[bs, sum(hw), 4]: filtered bbox proposals
        """
        N, S, C = memory.shape
        proposals = []
        _cur = 0  # start index of the ith layer
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur: (_cur + H * W)].view(N, H, W, 1)
            valid_H = ops.sum(~mask_flatten_[:, :, 0, 0], 1)  # bs
            valid_W = ops.sum(~mask_flatten_[:, 0, :, 0], 1)  # bs

            grid_y, grid_x = ops.meshgrid(
                ops.linspace(Tensor(0, dtype=ms.float32), Tensor(H - 1, dtype=ms.float32), H),
                ops.linspace(Tensor(0, dtype=ms.float32), Tensor(W - 1, dtype=ms.float32), W),
            )  # (h, w)

            grid = ops.cat([grid_x.expand_dims(-1), grid_y.expand_dims(-1)], -1)  # (h ,w, 2)

            scale = ops.cat([valid_W.expand_dims(-1), valid_H.expand_dims(-1)], 1).view(N, 1, 1, 2)
            # (bs, h ,w, 2), normalized to valid range
            grid = (grid.expand_dims(0).broadcast_to(N, -1, -1, -1) + 0.5) / scale
            wh = ops.ones_like(grid) * 0.05 * (2.0 ** lvl)  # preset wh, larger wh for higher level
            proposal = ops.cat((grid, wh), -1).view(N, -1, 4)  # (bs, hw, 4)
            proposals.append(proposal)
            _cur += H * W

        # filter proposal
        output_proposals = ops.cat(proposals, 1)  # (bs, sum(hw), 4)
        # filter those whose centers are too close to the margin or wh too small or too large
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = ops.log(output_proposals / (1 - output_proposals))  # unsigmoid
        # filter proposal in the padding area
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.expand_dims(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # also mask memory in the filtered position
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.expand_dims(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))  # channel-wise mlp
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        """Get the reference points of every pixel position of every level used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = ops.meshgrid(
                ops.linspace(Tensor(0.5, dtype=ms.float32),
                             Tensor(H - 0.5, dtype=ms.float32), H),
                ops.linspace(Tensor(0.5, dtype=ms.float32),
                             Tensor(W - 0.5, dtype=ms.float32), W),
            )  # (h, w)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)  # (bs, hw)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)  # (bs, hw)
            ref = ops.stack((ref_x, ref_y), -1)  # (bs, hw, 2)
            reference_points_list.append(ref)
        reference_points = ops.cat(reference_points_list, 1)  # (bs, sum(hw), 2)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # (bs sum(hw), nl, 2)
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid(non-pad) ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = ops.sum(~mask[:, :, 0], 1)
        valid_W = ops.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = ops.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


class DINOTransformerEncoder(TransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            post_norm: bool = False,
            num_feature_levels: int = 4,
    ):
        super(DINOTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def constuct(
            self,
            query,
            key,
            value,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs,
    ):

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DINOTransformerDecoder(TransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            return_intermediate: bool = True,
            num_feature_levels: int = 4,
            look_forward_twice=True,
    ):
        super(DINOTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        # values of bbox_embed and class_embed are set in outer class DINO
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

    def construct(
            self,
            query,
            key,
            value,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            reference_points=None,  # (bs, num_query, 4). normalized to the valid image area
            valid_ratios=None,  # (bs, num_level, 2)  non-pad area ratio in h and w direction
            **kwargs,
    ):
        """
            Returns:
                output (Tensor[bs, num_query, embed_dim]): output of each layer
                reference_points (Tensor[bs, num_query, 4|2]): output reference point of each layer
            """
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_query, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                # normalized to the whole image (including padding area)
                reference_points_input = (
                        reference_points[:, :, None]
                        * ops.cat([valid_ratios, valid_ratios], -1)[:, None]
                )  # (bs, num_query, num_level, 4)
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            # reference is almost the same for all level, pick the first one
            # TODO to compat with len(reference_points) == 2
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])  # (bs, num_query, embed_dim)
            query_pos = self.ref_point_head(query_sine_embed)  # (bs, num_query, embed_dim)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,  # list of masks for all attention layers
                query_key_padding_mask=query_key_padding_mask,  # key padding masks for self attention
                key_padding_mask=key_padding_mask,  # key padding masks for cross attention
                reference_points=reference_points_input,  # (bs, num_query, num_level, 4)
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    # both delta and refer will be supervised
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return ops.stack(intermediate), ops.stack(intermediate_reference_points)

        return output, reference_points
