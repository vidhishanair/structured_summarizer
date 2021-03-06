from __future__ import unicode_literals, print_function, division

import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.BilinearMatrixAttention import BilinearMatrixAttention
from models.modules.StructuredAttention import StructuredAttention
from models.modules.SemanticStrAttention import SemanticStrAttention
from models.model_utils import init_wt_normal
from utils import config
from numpy import random
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)


class StructuredEncoder(nn.Module):
    def __init__(self, args, vocab):
        super(StructuredEncoder, self).__init__()
        print("Using Structured Encoder")
        # self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)

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

        self.drop = nn.Dropout(0.3)
        init_wt_normal(self.embedding.weight)
        bidirectional = True
        # self.device = torch.device("cuda" if config.use_gpu else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args

        if bidirectional:
            self.sem_dim_size = 2*config.sem_dim_size
            self.sent_hidden_size = 2*config.hidden_dim
            self.doc_hidden_size = 2*config.hidden_dim
        else:
            self.sem_dim_size = config.sem_dim_size
            self.sent_hidden_size = config.hidden_dim
            self.doc_hidden_size = config.hidden_dim

        self.sentence_encoder = BiLSTMEncoder(self.device, self.sent_hidden_size, self.emb_dim, 1, dropout=0.3,
                                             bidirectional=bidirectional)
        self.document_encoder = BiLSTMEncoder(self.device, self.doc_hidden_size, self.sent_hidden_size, 1, dropout=0.3,
                                              bidirectional=bidirectional)
        self.sentence_structure_att = StructuredAttention(self.device, self.sem_dim_size, self.sent_hidden_size, bidirectional, "1.3.0")
        self.document_structure_att = StructuredAttention(self.device, self.sem_dim_size, self.doc_hidden_size, bidirectional, "1.3.0")
        self.sent_op_size = 0
        if not self.args.no_latent_str:
            self.sent_op_size += self.sem_dim_size
        if self.args.use_coref_att_encoder or self.args.no_latent_str:
            self.sem_structure_att = SemanticStrAttention(self.device, self.sem_dim_size, self.doc_hidden_size, bidirectional, "1.3.0")
            self.sent_op_size += self.sem_dim_size

        self.sent_pred_linear = nn.Linear(self.sent_op_size, 1)
        self.token_pred_linear = nn.Linear(self.sent_hidden_size, 2)
        self.doc_pred_linear = nn.Linear(self.sent_hidden_size+self.sent_op_size, 1)
        self.sm = nn.Softmax(dim=1)

        if args.heuristic_chains:
                self.bilinear = BilinearMatrixAttention(self.sent_op_size, self.sent_op_size, False, 1)
                self.sent_p_linear = nn.Linear(self.sent_op_size, self.sent_op_size)
                self.sent_c_linear = nn.Linear(self.sent_op_size, self.sent_op_size)
                self.sm2 = nn.Softmax(dim=2)
                self.bilinear_pall = BilinearMatrixAttention(self.sent_op_size, self.sent_op_size, True, self.sent_op_size)
                self.sent_p_linear_pall = nn.Linear(self.sent_op_size, self.sent_op_size)
                self.sent_c_linear_pall = nn.Linear(self.sent_op_size, self.sent_op_size)
                self.pred_linear_pall = nn.Linear(self.sent_op_size, 2)
                self.bilinear_call = BilinearMatrixAttention(self.sent_op_size, self.sent_op_size, True, self.sent_op_size)
                self.sent_p_linear_call = nn.Linear(self.sent_op_size, self.sent_op_size)
                self.sent_c_linear_call = nn.Linear(self.sent_op_size, self.sent_op_size)
                self.pred_linear_call = nn.Linear(self.sent_op_size, 2)


    #seq_lens should be in descending order
    def forward_test(self, input, sent_l, doc_l, tokens_mask, sent_mask, word_batch, word_padding_mask,
                     enc_word_lens, enc_tags_batch, enc_sent_token_mat, weighted_adj_mat):


        batch_size, sent_size, token_size = input.size()

        input = self.embedding(input)
        input = self.drop(input)
        word_input = self.embedding(word_batch) #
        word_input = self.drop(word_input)

        # BiLSTM
        bilstm_encoded_word_tokens, word_token_hidden = self.sentence_encoder.forward_packed(word_input, enc_word_lens)
        mask = word_padding_mask.unsqueeze(2).repeat(1, 1, self.sent_hidden_size)
        bilstm_encoded_word_tokens = bilstm_encoded_word_tokens * mask

        # reshape to 3D tensor
        input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))
        sent_l = Variable(torch.IntTensor(list(itertools.chain.from_iterable(sent_l))).int())

        # BiLSTM
        bilstm_encoded_tokens, token_hidden = self.sentence_encoder.forward_packed(input, sent_l)
        mask = tokens_mask.view(tokens_mask.size(0)*tokens_mask.size(1),
                                tokens_mask.size(2)).unsqueeze(2).repeat(1, 1, self.sent_hidden_size)
        bilstm_encoded_tokens = bilstm_encoded_tokens * mask

        bilstm_encoded_tokens = bilstm_encoded_tokens.contiguous().view(batch_size, sent_size, token_size, self.sent_hidden_size)
        masked_bilstm_encoded_tokens = bilstm_encoded_tokens + ((tokens_mask-1)*999).unsqueeze(3).repeat(1, 1, 1, self.sent_hidden_size)
        max_pooled_bilstm_sents = masked_bilstm_encoded_tokens.max(dim=2)[0]  # Batch * sent * dim
        #encoded_tokens = bilstm_encoded_tokens


        bilstm_encoded_sents, sent_hidden = self.document_encoder.forward_packed(max_pooled_bilstm_sents, doc_l)
        mask = sent_mask.unsqueeze(2).repeat(1,1, self.doc_hidden_size)
        bilstm_encoded_sents = bilstm_encoded_sents * mask
        # structure Att
        sa_encoded_sents = None
        sent_attention_matrix = torch.rand(batch_size, sent_size, 1+sent_size).cuda()
        if not self.args.no_latent_str:
            sa_encoded_sents, sent_attention_matrix = self.document_structure_att.forward(bilstm_encoded_sents)
            mask = sent_mask.unsqueeze(2).repeat(1,1, self.sem_dim_size)
            sa_encoded_sents = sa_encoded_sents * mask

        if self.args.use_coref_att_encoder or self.args.no_latent_str:
            sem_sa_encoded_sents = self.sem_structure_att.forward(bilstm_encoded_sents, weighted_adj_mat)
            mask = sent_mask.unsqueeze(2).repeat(1,1, self.sem_dim_size)
            sem_sa_encoded_sents = sem_sa_encoded_sents * mask
            if sa_encoded_sents is None:
                sa_encoded_sents = sem_sa_encoded_sents
            else:
                sa_encoded_sents = torch.cat([sa_encoded_sents, sem_sa_encoded_sents], dim=2)

        sa_encoded_sent_token_rep = torch.bmm(enc_sent_token_mat.permute(0,2,1).float(), sa_encoded_sents) # b * n_tokens * hid_dim

        # encoded_sents = sa_encoded_sents.unsqueeze(1).repeat(1, token_size, 1, 1).view(batch_size, sent_size*token_size,
        #                                                                                sa_encoded_sents.size(2))
        # # encoded_tokens = encoded_tokens.contiguous().view(batch_size, sent_size*token_size, encoded_tokens.size(3))
        # encoded_tokens = torch.cat([tk, encoded_sents], dim=2)
        encoded_tokens = torch.cat([bilstm_encoded_word_tokens, sa_encoded_sent_token_rep], dim=2)
        max_pooled_doc = encoded_tokens.max(dim=1)[0]


        mask = sent_mask.unsqueeze(1).repeat(1, sent_mask.size(1), 1) * sent_mask.unsqueeze(2) #.transpose(1,0)
        mask = torch.cat((sent_mask.unsqueeze(2), mask), dim=2)
        mat = sent_attention_matrix * mask
        sentence_importance_vector = mat[:,:,1:].sum(dim=1) #* sent_mask
        sentence_importance_vector = sentence_importance_vector / sentence_importance_vector.sum(dim=1, keepdim=True).repeat(1, sentence_importance_vector.size(1))
        token_level_sentence_scores = torch.bmm(enc_sent_token_mat.permute(0,2,1).float(), sentence_importance_vector.unsqueeze(2)).view(batch_size, enc_sent_token_mat.size(2))


        sent_score = self.sent_pred_linear(sa_encoded_sents)
        sent_score = sent_score * sent_mask.unsqueeze(2)
        sent_score = self.sm(sent_score)

        token_score = self.token_pred_linear(bilstm_encoded_word_tokens)
        mask = word_padding_mask.unsqueeze(2)
        token_score = token_score * mask

        doc_score = self.doc_pred_linear(max_pooled_doc)

        sent_single_head_scores = None
        sent_all_head_scores = None
        sent_all_child_scores = None

        if self.args.heuristic_chains:
            if self.args.use_sent_single_head_loss or self.args.predict_sent_single_head:
                sa_encoded_single_sents_p = F.tanh(self.sent_p_linear(sa_encoded_sents)) # bxsentxdim
                sa_encoded_single_sents_c = F.tanh(self.sent_c_linear(sa_encoded_sents)) # bxsentxdim
                sent_single_head_scores = self.bilinear(sa_encoded_single_sents_p, sa_encoded_single_sents_c).view(batch_size, sent_size, sent_size) #.squeeze() # b, sent , sent
                sent_single_head_scores = sent_single_head_scores * sent_mask.unsqueeze(1).repeat(1, sent_mask.size(1), 1)
                sent_single_head_scores = sent_single_head_scores * sent_mask.unsqueeze(2)

            if self.args.use_sent_all_head_loss or self.args.predict_sent_all_head:
                sa_encoded_all_sents_p = F.tanh(self.sent_p_linear_pall(sa_encoded_sents)) # bxsentxdim
                sa_encoded_all_sents_c = F.tanh(self.sent_c_linear_pall(sa_encoded_sents)) # bxsentxdim
                sent_all_head_scores = self.bilinear_pall(sa_encoded_all_sents_p, sa_encoded_all_sents_c).view(batch_size, sent_size, sent_size, self.sem_dim_size) #.squeeze() # b, sent , sent , dim
                sent_all_head_scores = self.pred_linear_pall(sent_all_head_scores) # b, sent, sent, 2
                sent_all_head_scores = sent_all_head_scores * sent_mask.unsqueeze(1).unsqueeze(3).repeat(1, sent_mask.size(1), 1, sent_all_head_scores.size(3))
                sent_all_head_scores = sent_all_head_scores * sent_mask.unsqueeze(2).unsqueeze(3)

            if self.args.use_sent_all_child_loss or self.args.predict_sent_all_child:
                sa_encoded_child_sents_c = F.tanh(self.sent_c_linear_call(sa_encoded_sents)) # bxsentxdim
                sa_encoded_child_sents_p = F.tanh(self.sent_p_linear_call(sa_encoded_sents)) # bxsentxdim
                sent_all_child_scores = self.bilinear_pall(sa_encoded_child_sents_c, sa_encoded_child_sents_p).view(batch_size, sent_size, sent_size, self.sem_dim_size) #.squeeze() # b, sent , sent , dim
                sent_all_child_scores = self.pred_linear_call(sent_all_child_scores) # b, sent, sent, 2
                sent_all_child_scores = sent_all_child_scores * sent_mask.unsqueeze(1).unsqueeze(3).repeat(1, sent_mask.size(1), 1, sent_all_child_scores.size(3))
                sent_all_child_scores = sent_all_child_scores * sent_mask.unsqueeze(2).unsqueeze(3)


        encoder_output = {"encoded_tokens": encoded_tokens,
                          "token_hidden": token_hidden,
                          "sentence_level_encoded_sents": sa_encoded_sents,
                          "encoded_sents": sa_encoded_sent_token_rep,
                          "sent_hidden": sent_hidden,
                          "document_rep" : max_pooled_doc,
                          "token_attention_matrix" : None,
                          "sent_attention_matrix" : sent_attention_matrix,
                          "sent_importance_vector" : sentence_importance_vector,
                          "token_level_sentence_scores" : token_level_sentence_scores,
                          "sent_score": sent_score,
                          "token_score": token_score,
                          "doc_score": doc_score,
                          "sent_single_head_scores": sent_single_head_scores,
                          "sent_all_head_scores": sent_all_head_scores,
                          "sent_all_child_scores": sent_all_child_scores}

        return encoder_output
