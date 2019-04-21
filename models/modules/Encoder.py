from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.StructuredAttention import StructuredAttention
from models.modules.StructuredEncoder import StructuredEncoder
from models.model_utils import init_wt_normal, init_lstm_wt, init_linear_wt, init_wt_unif
from utils import config
from numpy import random
import numpy as np
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        print("Using default encoder")
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.drop = nn.Dropout(0.3)
        self.device = torch.device("cuda" if config.use_gpu else "cpu")
        # self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # init_lstm_wt(self.lstm)

        self.bilstmenc = BiLSTMEncoder(self.device, config.hidden_dim, config.emb_dim, 1, dropout=0.3,
                                              bidirectional=True)


    #seq_lens should be in descending order
    def forward_withoutsorting(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        h, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        h = h.contiguous()
        max_h, _ = h.max(dim=1)

        return h, hidden, max_h

    def forward(self, input, sent_l, doc_l, mask_tokens, mask_sents):
        # Sort by length (keep idx)
        input = self.embedding(input)
        input = self.drop(input)
        seq_len = list(itertools.chain.from_iterable(sent_l))
        sent_output, hidden = self.bilstmenc(input, seq_len)
        max_h, _ = sent_output.max(dim=1)

        output = {"encoded_tokens": sent_output,
                  "sent_hidden": hidden,
                  "document_rep": max_h}

        return output
