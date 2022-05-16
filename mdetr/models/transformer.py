# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        contrastive_loss=False,
    ):
        super(Transformer, self).__init__()

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask=None,
        query_embed=None,
        pos_embed=None,
        pass_pos_and_query=True,
        text_memory_resized=None,
        text_attention_mask=None,
    ):

        if pass_pos_and_query:
            tgt = torch.zeros_like(query_embed)
        else:
            src, tgt, query_embed, pos_embed = (
                src + 0.1 * pos_embed,
                query_embed,
                None,
                None,
            )

        img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        text_memory = img_memory[-len(text_memory_resized) :]

        assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

        memory_cache = {
            "text_memory_resized": text_memory_resized,
            "text_memory": text_memory,
            "img_memory": img_memory,
            "mask": mask,
            "pos_embed": pos_embed,
            "query_embed": query_embed,
        }

        if pass_pos_and_query:
            tgt = torch.zeros_like(query_embed)
        else:
            src, tgt, query_embed, pos_embed = (
                src + 0.1 * pos_embed,
                query_embed,
                None,
                None,
            )

        hs = self.decoder(
            tgt,
            img_memory,
            text_memory,
            memory_key_padding_mask=mask,
            text_memory_key_padding_mask=text_attention_mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return memory_cache, hs.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src_attn = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src_attn)
        src = self.norm1(src)
        src_ffn = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src_ffn)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            src = self.norm1(src)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt_attn = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt_attn)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt_cross_attn = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt_cross_attn)
        tgt = self.norm3(tgt)

        # FFN
        tgt_ffn = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt_ffn)
        tgt = self.norm4(tgt)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            tgt = self.norm1(tgt)
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        contrastive_loss=args.contrastive_loss,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
