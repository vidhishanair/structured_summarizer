from __future__ import unicode_literals, print_function, division

import time
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
from models.model_utils import init_wt_normal, init_lstm_wt, init_linear_wt, init_wt_unif
from utils import config
from numpy import random
from utils.train_util import get_input_from_batch, get_output_from_batch
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if config.use_gpu else "cpu")

random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)


class Model(nn.Module):
    def __init__(self, args, vocab):
        super(Model, self).__init__()
        self.args = args
        if(args.fixed_scorer):
            pretrained_scorer = StructuredEncoder(args)
        encoder = StructuredEncoder(args, vocab)
        decoder = Decoder(args, vocab)
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight

        if(args.fixed_scorer):
            self.pretrained_scorer = pretrained_scorer
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if args.reload_pretrained_clf_path is not None and args.fixed_scorer:
            state = torch.load(args.reload_pretrained_clf_path, map_location= lambda storage, location: storage)
            self.pretrained_scorer.load_state_dict(state['encoder_state_dict'])
        elif args.reload_pretrained_clf_path is not None:
            state = torch.load(args.reload_pretrained_clf_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])

        if args.reload_path is not None:
            state = torch.load(args.reload_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'], strict=False)
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])

    def eval(self):
        if(self.args.fixed_scorer):
            self.pretrained_scorer = self.pretrained_scorer.eval()
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()
        self.reduce_state = self.reduce_state.eval()

    def train(self):
        if(self.args.fixed_scorer):
            self.pretrained_scorer = self.pretrained_scorer.train()
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()
        self.reduce_state = self.reduce_state.train()

    def get_app_outputs(self, encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab, enc_sent_token_mat):
        encoder_outputs = encoder_output["encoded_tokens"]
        enc_padding_mask = enc_padding_token_mask.contiguous().view(enc_padding_token_mask.size(0),
                                                                    enc_padding_token_mask.size(
                                                                        1) * enc_padding_token_mask.size(2))
        # enc_batch_extend_vocab = enc_batch_extend_vocab.contiguous().view(enc_batch_extend_vocab.size(0),
        #                                                                   enc_batch_extend_vocab.size(
        #                                                                       1) * enc_batch_extend_vocab.size(2))

        encoder_hidden = encoder_output["sent_hidden"]
        max_encoder_output = encoder_output["document_rep"]
        token_level_sentence_scores = encoder_output["token_level_sentence_scores"]
        sent_output = encoder_output['encoded_sents']
        token_scores = encoder_output['token_score']
        sent_scores = encoder_output['sent_score']
        sent_scores = torch.bmm(enc_sent_token_mat.permute(0,2,1).float(), sent_scores).view(sent_scores.size(0), enc_sent_token_mat.size(2))

        sent_attention_matrix = encoder_output['sent_attention_matrix']
        sent_level_rep = encoder_output['sentence_level_encoded_sents']

        return encoder_outputs, enc_padding_mask, encoder_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_output, token_scores, sent_scores, sent_attention_matrix, sent_level_rep

    def forward(self, enc_batch, enc_padding_token_mask, enc_padding_sent_mask,
                      enc_doc_lens, enc_sent_lens,
                      enc_batch_extend_vocab, extra_zeros, c_t_1, coverage,
                      word_batch, word_padding_mask, enc_word_lens, enc_tags_batch, enc_sent_token_mat,
                      max_dec_len, dec_batch, adj_mat, weighted_adj_mat, undir_weighted_adj_mat, args):

        start = time.time()
        enc_adj_mat = adj_mat
        if args.use_weighted_annotations:
            if args.use_undirected_weighted_graphs:
                enc_adj_mat = undir_weighted_adj_mat
            else:
                enc_adj_mat = weighted_adj_mat
        encoder_output = self.encoder.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask,
                                                   enc_padding_sent_mask, word_batch, word_padding_mask,
                                                   enc_word_lens, enc_tags_batch, enc_sent_token_mat, enc_adj_mat)
        #print('Time taken for encoder: ', time.time() - start)

        encoder_outputs, enc_padding_mask, encoder_last_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_outputs, token_scores, sent_scores, sent_attention_matrix, sent_level_rep = \
            self.get_app_outputs(encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab, enc_sent_token_mat)

        if(args.fixed_scorer):
            scorer_output = self.model.module.pretrained_scorer.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask, enc_padding_sent_mask, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch)
            token_scores = scorer_output['token_score']
            sent_scores = scorer_output['sent_score'].unsqueeze(1).repeat(1, enc_padding_token_mask.size(2),1, 1).view(enc_padding_token_mask.size(0), enc_padding_token_mask.size(1)*enc_padding_token_mask.size(2))

        all_child, all_head = None, None
        if args.use_gold_annotations_for_decode:
            if args.use_weighted_annotations:
                if args.use_undirected_weighted_graphs:
                    permuted_all_head = undir_weighted_adj_mat[:, :, :].permute(0,2,1)
                    all_head = permuted_all_head.clone()
                    row_sums = torch.sum(permuted_all_head, dim=2, keepdim=True)
                    all_head[row_sums.expand_as(permuted_all_head)!=0] = permuted_all_head[row_sums.expand_as(permuted_all_head)!=0]/row_sums.expand_as(permuted_all_head)[row_sums.expand_as(permuted_all_head)!=0]

                    base_all_child = undir_weighted_adj_mat[:, :, :]
                    all_child = base_all_child.clone()
                    row_sums = torch.sum(base_all_child, dim=2, keepdim=True)
                    all_child[row_sums.expand_as(base_all_child)!=0] = base_all_child[row_sums.expand_as(base_all_child)!=0]/row_sums.expand_as(base_all_child)[row_sums.expand_as(base_all_child)!=0]
                else:
                    permuted_all_head = weighted_adj_mat[:, :, :].permute(0,2,1)
                    all_head = permuted_all_head.clone()
                    row_sums = torch.sum(permuted_all_head, dim=2, keepdim=True)
                    all_head[row_sums.expand_as(permuted_all_head)!=0] = permuted_all_head[row_sums.expand_as(permuted_all_head)!=0]/row_sums.expand_as(permuted_all_head)[row_sums.expand_as(permuted_all_head)!=0]

                    base_all_child = weighted_adj_mat[:, :, :]
                    all_child = base_all_child.clone()
                    row_sums = torch.sum(base_all_child, dim=2, keepdim=True)
                    all_child[row_sums.expand_as(base_all_child)!=0] = base_all_child[row_sums.expand_as(base_all_child)!=0]/row_sums.expand_as(base_all_child)[row_sums.expand_as(base_all_child)!=0]
            else:
                permuted_all_head = adj_mat[:, :, :].permute(0,2,1)
                all_head = permuted_all_head.clone()
                row_sums = torch.sum(permuted_all_head, dim=2, keepdim=True)
                all_head[row_sums.expand_as(permuted_all_head)!=0] = permuted_all_head[row_sums.expand_as(permuted_all_head)!=0]/row_sums.expand_as(permuted_all_head)[row_sums.expand_as(permuted_all_head)!=0]
                
                base_all_child = adj_mat[:, :, :]
                all_child = base_all_child.clone()
                row_sums = torch.sum(base_all_child, dim=2, keepdim=True)
                all_child[row_sums.expand_as(base_all_child)!=0] = base_all_child[row_sums.expand_as(base_all_child)!=0]/row_sums.expand_as(base_all_child)[row_sums.expand_as(base_all_child)!=0]
                # all_head = adj_mat[:, :, :].permute(0,2,1) + 0.00005
                # row_sums = torch.sum(all_head, dim=2, keepdim=True)
                # all_head = all_head / row_sums
                # all_child = adj_mat[:, :, :] + 0.00005
                # row_sums = torch.sum(all_child, dim=2, keepdim=True)
                # all_child = all_child / row_sums


        s_t_1 = self.reduce_state(encoder_last_hidden)
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output

        final_dist_list = []
        attn_dist_list = []
        p_gen_list = []
        coverage_list = []
        start = time.process_time()
        if args.use_summ_loss:
            for di in range(min(max_dec_len, self.args.max_dec_steps)):
                y_t_1 = dec_batch[:, di]  # Teacher forcing
                final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage = self.decoder.forward(y_t_1, s_t_1,
                                                                                            encoder_outputs,
                                                                                            word_padding_mask, c_t_1,
                                                                                            extra_zeros,
                                                                                            enc_batch_extend_vocab,
                                                                                            coverage, token_scores,
                                                                                            sent_scores, sent_outputs,
                                                                                            enc_sent_token_mat,
                                                                                            all_head, all_child,
                                                                                            sent_level_rep)
                final_dist_list.append(final_dist)
                attn_dist_list.append(attn_dist)
                p_gen_list.append(p_gen)
                coverage_list.append(coverage)
            final_dist_list = torch.stack(final_dist_list, dim=1)
            attn_dist_list = torch.stack(attn_dist_list, dim=1)
            p_gen_list = torch.stack(p_gen_list, dim=1)
            if self.args.is_coverage:
                coverage_list = torch.stack(coverage_list, dim=1)
        #print('Time taken for decoder: ', time.process_time() - start)
        # print(coverage_list)
        # return torch.stack(final_dist_list, dim=1), torch.stack(attn_dist_list, dim=1), torch.stack(p_gen_list, dim=1), 
        return final_dist_list, attn_dist_list, p_gen_list, coverage_list, sent_attention_matrix, encoder_output['sent_single_head_scores'], \
               encoder_output['sent_all_head_scores'], encoder_output['sent_all_child_scores'], \
               encoder_output['token_score'], encoder_output['sent_score'], encoder_output['doc_score'],



