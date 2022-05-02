# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .backbone import build_backbone, build_text_backbone


class MDETR(nn.Module):
    """This is the MDETR module that performs modulated object detection"""

    def __init__(self, backbone_vision, roberta):
        super().__init__()
        self.backbone_vision = backbone_vision
        self.roberta = roberta


def build_model(args):
    backbone_vison = build_backbone(args)
    roberta = build_text_backbone(args)
    model = MDETR(backbone_vision, roberta)
    return model
