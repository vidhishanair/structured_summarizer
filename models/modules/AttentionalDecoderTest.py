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
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        hidden_reduced_h = F.relu(self.reduce_h(h.view(-1, config.hidden_dim * 2)))
        hidden_reduced_c = F.relu(self.reduce_c(c.view(-1, config.hidden_dim * 2)))

        return hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0) # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        # attention
        self.concat_rep = args.concat_rep
        self.is_coverage = args.is_coverage
        self.no_sent_sa = args.no_sent_sa
        self.args = args
        if self.args.concat_rep:
            self.encoder_op_size = config.sem_dim_size * 2 + config.hidden_dim * 2
        else:
            self.encoder_op_size = config.hidden_dim * 2
        self.W_h = nn.Linear(self.encoder_op_size, config.hidden_dim * 2, bias=False)
        if self.args.sep_sent_features:
            self.W_s = nn.Linear(2*config.sem_dim_size, config.hidden_dim * 2, bias=False)


        if self.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, h, enc_padding_mask, coverage, token_level_sentence_scores, s):
        b, t_k, n1 = list(h.size())
        h = h.view(-1, n1)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(h)
        a, n = list(encoder_feature.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim

        if self.args.sep_sent_features:
            #s = s.view(-1, n1)
            sent_features = self.W_s(s)
            sent_features = sent_features.view(-1,n)
            att_features = att_features + sent_features

        if self.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        if (self.args.gold_tag_scores and self.training) or self.args.decode_setting:
            scores = scores * token_level_sentence_scores
        elif self.args.sent_score_decoder:
            scores = scores + token_level_sentence_scores
        else:
            scores = scores

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        h = h.view(-1, t_k, n1)  # B x t_k x 2*hidden_dim
        c_t = torch.bmm(attn_dist, h)  # B x 1 x n
        c_t = c_t.view(-1, self.encoder_op_size)

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.attention_network = Attention(args)
        self.pointer_gen = args.pointer_gen
        self.args = args
        if self.args.concat_rep:
            self.encoder_op_size = config.sem_dim_size * 2 + config.hidden_dim * 2
        else:
            self.encoder_op_size = config.hidden_dim * 2
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.x_context = nn.Linear(self.encoder_op_size + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.p_gen_linear = nn.Linear(config.hidden_dim * 2 + self.encoder_op_size + config.emb_dim, 1)
        self.out1 = nn.Linear(config.hidden_dim + self.encoder_op_size, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)

        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, token_level_sentence_scores, sent_features):

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage = self.attention_network(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage, token_level_sentence_scores, sent_features)
        p_gen = None
        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
