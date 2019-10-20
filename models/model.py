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
from models.model_utils import init_wt_normal, init_lstm_wt, init_linear_wt, init_wt_unif
from utils import config
from numpy import random
from utils.train_util import get_input_from_batch, get_output_from_batch
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if config.use_gpu else "cpu")

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        if(args.fixed_scorer):
            pretrained_scorer = StructuredEncoder(args)
        encoder = StructuredEncoder(args)
        decoder = Decoder(args)
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight

        # if use_cuda:
        #     if(args.fixed_scorer):
        #         pretrained_scorer = pretrained_scorer.to(device)
        #     encoder = encoder.to(device)
        #     decoder = decoder.to(device)
        #     reduce_state = reduce_state.to(device)

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
            self.encoder.load_state_dict(state['encoder_state_dict'])
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

    def get_app_outputs(self, encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab):
        encoder_outputs = encoder_output["encoded_tokens"]
        enc_padding_mask = enc_padding_token_mask.contiguous().view(enc_padding_token_mask.size(0),
                                                                    enc_padding_token_mask.size(
                                                                        1) * enc_padding_token_mask.size(2))
        enc_batch_extend_vocab = enc_batch_extend_vocab.contiguous().view(enc_batch_extend_vocab.size(0),
                                                                          enc_batch_extend_vocab.size(
                                                                              1) * enc_batch_extend_vocab.size(2))
        # else:
        #     encoder_outputs = encoder_output["encoded_sents"]
        #     enc_padding_mask = enc_padding_sent_mask

        encoder_hidden = encoder_output["sent_hidden"]
        max_encoder_output = encoder_output["document_rep"]
        token_level_sentence_scores = encoder_output["token_level_sentence_scores"]
        sent_output = encoder_output['encoded_sents']
        token_scores = encoder_output['token_score']
        sent_scores = encoder_output['sent_score'].unsqueeze(2).repeat(1,1, enc_padding_token_mask.size(2), 1).view(enc_padding_token_mask.size(0), enc_padding_token_mask.size(1)*enc_padding_token_mask.size(2))
        return encoder_outputs, enc_padding_mask, encoder_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_output, token_scores, sent_scores

    def forward(self, enc_batch, enc_padding_token_mask, enc_padding_sent_mask,
                      enc_doc_lens, enc_sent_lens,
                      enc_batch_extend_vocab, extra_zeros, c_t_1, coverage,
                      word_batch, word_padding_mask, enc_word_lens, enc_tags_batch,
                      max_dec_len, dec_batch, args):
        #
        # dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
        #     get_output_from_batch(batch, use_cuda)

        encoder_output = self.encoder.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask, enc_padding_sent_mask, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch)

        encoder_outputs, enc_padding_mask, encoder_last_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_outputs, token_scores, sent_scores = \
            self.get_app_outputs(encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab)

        if(args.fixed_scorer):
            scorer_output = self.model.module.pretrained_scorer.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask, enc_padding_sent_mask, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch)
            token_scores = scorer_output['token_score']
            sent_scores = scorer_output['sent_score'].unsqueeze(1).repeat(1, enc_padding_token_mask.size(2),1, 1).view(enc_padding_token_mask.size(0), enc_padding_token_mask.size(1)*enc_padding_token_mask.size(2))


        s_t_1 = self.reduce_state(encoder_last_hidden)
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output

        final_dist_list = []
        attn_dist_list = []
        p_gen_list = []
        coverage_list = []

        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage = self.decoder.forward(y_t_1, s_t_1,
                                                                                        encoder_outputs,
                                                                                        enc_padding_mask, c_t_1,
                                                                                        extra_zeros,
                                                                                        enc_batch_extend_vocab,
                                                                                        coverage, token_scores,
                                                                                        sent_scores, sent_outputs)
            final_dist_list.append(final_dist)
            attn_dist_list.append(attn_dist)
            p_gen_list.append(p_gen)
            coverage_list.append(coverage)

        return torch.stack(final_dist_list, dim=1), torch.stack(attn_dist, dim=1), \
               torch.stack(p_gen_list, dim=1), torch.stack(coverage_list, dim=1)



