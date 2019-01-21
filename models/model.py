from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#from models.modules.AttentionalDecoder import Decoder, ReduceState
from models.modules.AttentionalDecoderTest import Decoder, ReduceState
from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.Encoder import Encoder
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


class Model(object):
    def __init__(self, args):
        if args.no_sa:
            encoder = Encoder(args)
        else:
            encoder = StructuredEncoder(args)
        decoder = Decoder(args)
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if args.reload_path is not None:
            state = torch.load(args.reload_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def eval(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()
        self.reduce_state = self.reduce_state.eval()

    def train(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()
        self.reduce_state = self.reduce_state.train()
