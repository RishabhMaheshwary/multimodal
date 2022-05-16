# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtext.functional as F
import torchtext.transforms as T
from torch.nn import Module


class RobertaTransform(Module):
    def __init__(self):
        super().__init__()
        # Instantiate various transforms

        # Tokenizer to split input text into tokens
        encoder_json_path = (
            "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
        )
        vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
        self.tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)

        # vocabulary converting tokens to IDs
        vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
        self.vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))

        # Add BOS token to the beginning of sentence
        self.add_bos = T.AddToken(token=0, begin=True)

        # Add EOS token to the end of sentence
        self.add_eos = T.AddToken(token=2, begin=False)

    def forward(self, input):
        tokens = self.tokenizer(input["text"])
        tokens = F.truncate(tokens, max_seq_len=254)
        tokens = self.vocab(tokens)
        tokens = self.add_bos(tokens)
        tokens = self.add_eos(tokens)
        input["tokens"] = tokens
        return input
