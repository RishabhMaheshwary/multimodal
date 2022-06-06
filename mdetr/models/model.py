# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaModel, RobertaTokenizerFast
from utils.misc import NestedTensor

from .backbone import build_backbone, build_text_backbone
from .losses import ContrastiveCriterion, MdetrPretrainingLosses
from .matcher import build_matcher
from .transformer import build_transformer


class MDETR(nn.Module):
    """This is the MDETR module that performs modulated object detection"""

    def __init__(
        self,
        backbone_vision,
        roberta,
        transformer,
        d_model=512,
        text_encoder_type="roberta-base",
        num_classes=255,
        num_queries=100,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        contrastive_align_loss=False,
        qa_dataset: Optional[str] = None,
        split_qa_heads=True,
        predict_final=False,
    ):

        super().__init__()
        self.backbone = backbone_vision
        self.d_model = d_model
        hidden_dim = transformer.d_model
        # self.text_encoder = roberta
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.aux_loss = aux_loss
        self.expander_dropout = 0.1
        self.resizer = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )
        self.transformer = transformer
        self.CLS = nn.Embedding(1, d_model) if contrastive_loss is not None else None
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, hidden_dim, kernel_size=1
        )

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.contrastive_loss = contrastive_loss
        if contrastive_loss is not None:
            self.contrastive_projection_image = nn.Linear(
                hidden_dim, contrastive_hdim, bias=False
            )
            self.contrastive_projection_text = nn.Linear(
                self.text_encoder.config.hidden_size,
                contrastive_hdim,
                bias=False,
            )
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss is not None:
            self.contrastive_align_projection_image = nn.Linear(
                hidden_dim, contrastive_hdim
            )
            self.contrastive_align_projection_text = nn.Linear(
                hidden_dim, contrastive_hdim
            )

    def forward(
        self,
        samples,
        targets,
        positive_map,
        captions,
        encode_and_save=True,
        memory_cache=None,
        training=True,
    ):

        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            query_embed = self.query_embed.weight
            pos_embed = pos[-1]

            src = self.input_proj(src)
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            device = src.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            mask = mask.flatten(1)
            if self.CLS is not None:

                CLS = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)

                src = torch.cat((CLS, src))

                pos_embed = torch.cat(
                    (torch.zeros(1, bs, self.d_model, device=device), pos_embed)
                )

                cls_pad = torch.zeros(bs, 1).bool().to(device)
                mask = torch.cat((cls_pad, mask), dim=1)

            tokenized = self.tokenizer.batch_encode_plus(
                captions, padding="longest", return_tensors="pt"
            ).to(device)

            encoded_text = self.text_encoder(**tokenized)

            text_memory = encoded_text.last_hidden_state.transpose(0, 1)

            text_attention_mask = tokenized.attention_mask.ne(1).bool()

            text_memory_resized = self.resizer(text_memory)

            src = torch.cat([src, text_memory_resized], dim=0)

            mask = torch.cat([mask, text_attention_mask], dim=1)

            pos_embed = torch.cat(
                [pos_embed, torch.zeros_like(text_memory_resized)], dim=0
            )
            memory_cache, hs = self.transformer(
                src,
                mask,
                query_embed,
                pos_embed,
                text_memory_resized=text_memory_resized,
                text_attention_mask=text_attention_mask,
            )
            memory_cache.update(
                {
                    "text_attention_mask": text_attention_mask,
                    "tokenized": tokenized,
                    "text_pooled_op": encoded_text.pooler_output
                    if self.CLS is not None
                    else None,
                }
            )
            memory_cache.update(
                {
                    "img_pooled_op": memory_cache["img_memory"][0]
                    if self.CLS is not None
                    else None,  # Return the CLS token
                }
            )
            if self.contrastive_loss:
                memory_cache["text_pooled_op"] = self.contrastive_projection_text(
                    memory_cache["text_pooled_op"]
                )
                memory_cache["img_pooled_op"] = self.contrastive_projection_image(
                    memory_cache["img_pooled_op"]
                )

            out = {}
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                }
            )
            outputs_isfinal = None
            if self.isfinal_embed is not None:
                outputs_isfinal = self.isfinal_embed(hs)
                out["pred_isfinal"] = outputs_isfinal[-1]
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(
                    self.contrastive_align_projection_image(hs), p=2, dim=-1
                )
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(
                        memory_cache["text_memory"]
                    ).transpose(0, 1),
                    p=2,
                    dim=-1,
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )
            if self.aux_loss:
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c in zip(
                            outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1]
                        )
                    ]
                else:
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                        }
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                    ]
                if outputs_isfinal is not None:
                    assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                    for i in range(len(outputs_isfinal[:-1])):
                        out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]

            return out, memory_cache


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def build_model(args):
    backbone_vision = build_backbone(args)
    roberta = build_text_backbone(args)
    transformer = build_transformer(args)
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef}
    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.predict_final:
        weight_dict["loss_isfinal"] = 1

    weight_dict["loss_giou"] = args.giou_loss_coef
    losses_to_calculate = ["labels", "boxes", "cardinality"]
    if args.contrastive_align_loss:
        losses_to_calculate += ["contrastive_align"]

    losses = MdetrPretrainingLosses(
        matcher=matcher,
        eos_coef=args.eos_coef,
        losses=losses_to_calculate,
        temperature=args.temperature_NCE,
    )
    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE)
        contrastive_criterion.to(device)
    else:
        contrastive_criterion = None
    losses.to(torch.device(args.device))
    model = MDETR(
        backbone_vision,
        roberta,
        transformer,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        text_encoder_type=args.text_encoder_type,
        aux_loss=args.aux_loss,
        d_model=args.hidden_dim,
    )
    return model, losses, contrastive_criterion, weight_dict
