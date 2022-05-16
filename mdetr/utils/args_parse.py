# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def get_args_parser():

    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=512,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--pre_norm", action="store_true")

    parser.add_argument("--output_dir", default=None, required=True)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)

    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--masks", action="store_true")
    parser.add_argument("--vg_img_path", type=str, default="")
    parser.add_argument(
        "--freeze_text_encoder",
        action="store_true",
        help="Whether to freeze the weights of the text encoder",
    )
    parser.add_argument("--text_encoder_type", type=str, default="roberta-base")
    parser.add_argument("--vg_ann_path", type=str, default="")
    parser.add_argument("--clevr_img_path", type=str, default="")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to run evaluation on val or test set",
    )
    parser.add_argument("--clevr_ann_path", type=str, default="")
    parser.add_argument("--phrasecut_ann_path", type=str, default="")
    parser.add_argument(
        "--phrasecut_orig_ann_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--contrastive_loss",
        action="store_true",
        help="Whether to add contrastive loss",
    )
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    return parser
