from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.StructuredAttention import StructuredAttention
from models.modules.StructuredEncoder import StructuredEncoder
from models.utils import init_wt_normal, init_lstm_wt, init_linear_wt, init_wt_unif
from utils import config
from numpy import random
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        h, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        h = h.contiguous()
        max_h, _ = h.max(dim=1)

        return h, hidden, max_h