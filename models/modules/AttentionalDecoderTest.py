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
        self.is_coverage = args.is_coverage
        self.args = args
        self.encoder_op_size = config.sem_dim_size * 2 + config.hidden_dim * 2
        self.W_h = nn.Linear(self.encoder_op_size, config.hidden_dim * 2, bias=False)
        if self.args.sep_sent_features:
            self.W_s = nn.Linear(2*config.sem_dim_size, config.hidden_dim * 2, bias=False)


        if self.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        self.v2 = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, h, enc_padding_mask, coverage, token_scores, sent_scores, s, enc_sent_token_mat, sent_all_head_scores, sent_all_child_scores, sent_level_rep):
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
        scores = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        attn_dist_ = scores.clone()
        # if self.args.token_scores:
        #     #print(scores.size())
        #     #print(token_scores.size())
        #     scores = scores * token_scores[:,:,1]
        # if self.args.sent_scores:
        #     scores = scores * sent_scores

        if self.args.sent_attention_at_dec:
            bs, n_s, hsize = sent_level_rep.size()
            current_dec_sent_rep = dec_fea.unsqueeze(1).expand(b, n_s, hsize).contiguous()
            sent_features = F.tanh(self.W_s(sent_level_rep + current_dec_sent_rep)) # B x n_s x 2*hdim
            sent_scores = self.v2(sent_features) # B x n_s x 1
            token_level_scores = torch.bmm(enc_sent_token_mat.permute(0,2,1), sent_scores) # B x tk x 1
            token_level_scores = token_level_scores.view(-1, t_k)
            attn_dist_ *= F.softmax(token_level_scores, dim=1)*enc_padding_mask


        if self.args.use_all_sent_head_at_decode:
            sent_att_scores = torch.bmm(enc_sent_token_mat, scores.unsqueeze(2)) # B x n_s x 1
            new_attended_sent_scores = torch.bmm(sent_att_scores.permute(0,2,1), sent_all_head_scores).permute(0,2,1) # B x n_s x 1
            new_head_token_scores = torch.bmm(enc_sent_token_mat.permute(0,2,1),
                                              new_attended_sent_scores).view(scores.size(0), scores.size(1))
            new_head_token_scores = F.softmax(new_head_token_scores, dim=1)*enc_padding_mask
            attn_dist_ += new_head_token_scores # to add to attention, need to test multiplication
        if self.args.use_all_sent_child_at_decode:
            sent_att_scores = torch.bmm(enc_sent_token_mat, scores.unsqueeze(2)) # B x n_s x 1
            new_attended_sent_scores = torch.bmm(sent_att_scores.permute(0,2,1), sent_all_child_scores).permute(0,2,1) # B x n_s x 1
            new_child_token_scores = torch.bmm(enc_sent_token_mat.permute(0,2,1),
                                               new_attended_sent_scores).view(scores.size(0), scores.size(1))
            attn_dist_ += F.softmax(new_child_token_scores, dim=1)*enc_padding_mask
        if self.args.use_single_sent_head_at_decode:
            print("Not Implemented for single_sent_head in decode")
            exit()

        # attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        #print(attn_dist)
        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        h = h.view(-1, t_k, n1)  # B x t_k x 2*hidden_dim
        c_t = torch.bmm(attn_dist, h)  # B x 1 x n
        c_t = c_t.view(-1, self.encoder_op_size)

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.is_coverage or self.args.bu_coverage_penalty:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, args, vocab):
        super(Decoder, self).__init__()
        self.attention_network = Attention(args)
        self.pointer_gen = args.pointer_gen
        self.args = args
        self.encoder_op_size = config.sem_dim_size * 2 + config.hidden_dim * 2
        # decoder
        # self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        # init_wt_normal(self.embedding.weight)
        
        if not args.use_glove:
            print("Using Random normal initialization for embeddings")
            self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
            init_wt_normal(self.embedding.weight)
            self.emb_dim = config.emb_dim
        else:
            print("Using Pre-trained embeddings")
            emb_tensor = torch.from_numpy(vocab.embedding_matrix)
            self.embedding = nn.Embedding.from_pretrained(emb_tensor)
            self.emb_dim = emb_tensor.size(1)        
  
        self.x_context = nn.Linear(self.encoder_op_size + self.emb_dim, self.emb_dim)

        self.lstm = nn.LSTM(self.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        self.p_gen_linear = nn.Linear(config.hidden_dim * 2 + self.encoder_op_size + self.emb_dim, 1)
        self.out1 = nn.Linear(config.hidden_dim + self.encoder_op_size, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)

        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, token_scores, sent_scores, sent_features, enc_sent_token_mat, sent_all_head_scores, sent_all_child_scores, sent_level_rep):

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage = self.attention_network(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage, token_scores, sent_scores,
                                                          sent_features, enc_sent_token_mat,
                                                          sent_all_head_scores, sent_all_child_scores, sent_level_rep)
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
