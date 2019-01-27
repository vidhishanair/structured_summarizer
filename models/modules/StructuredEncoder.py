from __future__ import unicode_literals, print_function, division

import time

import torch
import torch.nn as nn

from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.StructuredAttention import StructuredAttention
from models.utils import init_wt_normal
from utils import config
from numpy import random
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class StructuredEncoder(nn.Module):
    def __init__(self, args):
        super(StructuredEncoder, self).__init__()
        print("Using Structured Encoder")
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.concat_rep = args.concat_rep
        self.drop = nn.Dropout(0.3)
        init_wt_normal(self.embedding.weight)
        bidirectional = True
        device = torch.device("cuda" if config.use_gpu else "cpu")
        self.no_sent_sa = args.no_sent_sa

        #self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        if bidirectional:
            self.sem_dim_size = 2*config.sem_dim_size
            self.sent_hidden_size = 2*config.hidden_dim
            self.doc_hidden_size = 2*config.hidden_dim
        else:
            self.sem_dim_size = config.sem_dim_size
            self.sent_hidden_size = config.hidden_dim
            self.doc_hidden_size = config.hidden_dim

        self.sentence_encoder = BiLSTMEncoder(device, self.sent_hidden_size, config.emb_dim, 1, dropout=0.3,
                                              bidirectional=bidirectional)
        if args.no_sent_sa:
            self.document_encoder = BiLSTMEncoder(device, self.doc_hidden_size, self.sent_hidden_size, 1, dropout=0.3,
                                              bidirectional=bidirectional)
        else:
            self.document_encoder = BiLSTMEncoder(device, self.doc_hidden_size, self.sem_dim_size, 1, dropout=0.3,
                                                  bidirectional=bidirectional)

        self.sentence_structure_att = StructuredAttention(device, self.sem_dim_size, self.sent_hidden_size, bidirectional, "nightly")
        self.document_structure_att = StructuredAttention(device, self.sem_dim_size, self.doc_hidden_size, bidirectional, "nightly")

        #init_lstm_wt(self.sentence_encoder)
        #init_lstm_wt()

    #seq_lens should be in descending order
    def forward(self, input, sent_l, doc_l, mask_tokens, mask_sents):

        batch_size, sent_size, token_size = input.size()

        tokens_mask = mask_tokens
        sent_mask = mask_sents

        input = self.embedding(input)
        input = self.drop(input)

        # reshape to 3D tensor
        input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))
        sent_l = list(itertools.chain.from_iterable(sent_l))


        # BiLSTM
        bilstm_encoded_tokens, token_hidden = self.sentence_encoder.forward_packed(input, sent_l)
        mask = tokens_mask.view(tokens_mask.size(0)*tokens_mask.size(1),
                                tokens_mask.size(2)).unsqueeze(2).repeat(1, 1, self.sent_hidden_size)
        bilstm_encoded_tokens = bilstm_encoded_tokens * mask

        # Structure ATT
        sa_encoded_tokens, token_attention_matrix = self.sentence_structure_att.forward(bilstm_encoded_tokens)

        # Reshape and max pool
        sa_encoded_tokens = sa_encoded_tokens.contiguous().view(batch_size, sent_size, token_size, self.sem_dim_size)
        masked_sa_encoded_tokens = sa_encoded_tokens + ((tokens_mask-1)*999).unsqueeze(3).repeat(1, 1, 1, self.sem_dim_size)
        max_pooled_sa_sents = masked_sa_encoded_tokens.max(dim=2)[0]  # Batch * sent * dim
        bilstm_encoded_tokens = bilstm_encoded_tokens.contiguous().view(batch_size, sent_size, token_size, self.sent_hidden_size)
        masked_bilstm_encoded_tokens = bilstm_encoded_tokens + ((tokens_mask-1)*999).unsqueeze(3).repeat(1, 1, 1, self.sent_hidden_size)
        max_pooled_bilstm_sents = masked_bilstm_encoded_tokens.max(dim=2)[0]  # Batch * sent * dim

        # Doc BiLSTM
        if self.no_sent_sa:
            encoded_tokens =  bilstm_encoded_tokens
            bilstm_encoded_sents, sent_hidden = self.document_encoder.forward_packed(max_pooled_bilstm_sents, doc_l)
        else:
            encoded_tokens = sa_encoded_tokens
            bilstm_encoded_sents, sent_hidden = self.document_encoder.forward_packed(max_pooled_sa_sents, doc_l)

        mask = sent_mask.unsqueeze(2).repeat(1,1, self.doc_hidden_size)
        bilstm_encoded_sents = bilstm_encoded_sents * mask

        # structure Att
        sa_encoded_sents, sent_attention_matrix = self.document_structure_att.forward(bilstm_encoded_sents)

        # Max Pool
        masked_sa_encoded_sents = sa_encoded_sents + ((sent_mask-1)*999).unsqueeze(2).repeat(1,1, self.sem_dim_size)
        max_pooled_sa_doc = masked_sa_encoded_sents.max(dim=1)[0] #Batch * dim
        mask = sent_mask.unsqueeze(2).repeat(1,1, self.sem_dim_size)
        sa_encoded_sents = sa_encoded_sents * mask

        if self.concat_rep:
            # ext_encoded_documents = orig_encoded_documents.contiguous().view(orig_encoded_documents.size(0)*orig_encoded_documents.size(1), orig_encoded_documents.size(2))
            encoded_sents = sa_encoded_sents.unsqueeze(1).repeat(1, token_size, 1, 1).view(batch_size, sent_size*token_size,
                                                                                        sa_encoded_sents.size(2))
            encoded_tokens = encoded_tokens.contiguous().view(batch_size, sent_size*token_size, encoded_tokens.size(3))
            encoded_tokens = torch.cat([encoded_tokens, encoded_sents], dim=2)
            max_pooled_doc = encoded_tokens.max(dim=1)[0]
        else:
            encoded_tokens = encoded_tokens.contiguous().view(batch_size, sent_size*token_size, encoded_tokens.size(3))

        encoder_output = {"encoded_tokens": encoded_tokens,
                          "token_hidden": token_hidden,
                          "encoded_sents": sa_encoded_sents,
                          "sent_hidden": sent_hidden,
                          "document_rep" : max_pooled_doc,
                          "token_attention_matrix" : token_attention_matrix,
                          "sent_attention_matrix" : sent_attention_matrix}

        return encoder_output

    #seq_lens should be in descending order
    def forward_test(self, input, sent_l, doc_l, mask_tokens, mask_sents, word_batch, word_padding_mask, enc_word_lens):

        
        batch_size, sent_size, token_size = input.size()

        tokens_mask = mask_tokens
        sent_mask = mask_sents

        input = self.embedding(input)
        input = self.drop(input)

        word_input = self.embedding(word_batch)
        word_input = self.drop(word_input)
        # BiLSTM
        bilstm_encoded_word_tokens, word_token_hidden = self.sentence_encoder.forward_packed(word_input, enc_word_lens)
        mask = word_padding_mask.unsqueeze(2).repeat(1, 1, self.sent_hidden_size)
        bilstm_encoded_word_tokens = bilstm_encoded_word_tokens * mask

        tk = torch.zeros(batch_size, sent_size*token_size, bilstm_encoded_word_tokens.size(2)).cuda()

        for i in range(len(sent_l)):
            start_count = 0
            start_count2 = 0
            max_l = max(list(itertools.chain.from_iterable(sent_l)))
            for l in sent_l[i]:
                #print(i, start_count, l, start_count2, max_l)
                if l > 0:
                    tk[i, start_count:start_count+l,:] = bilstm_encoded_word_tokens[i,start_count2:start_count2+l,:]
                start_count = start_count+max_l
                start_count2 = start_count2+l

        # reshape to 3D tensor
        input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))
        sent_l = list(itertools.chain.from_iterable(sent_l))


        # BiLSTM
        bilstm_encoded_tokens, token_hidden = self.sentence_encoder.forward_packed(input, sent_l)
        mask = tokens_mask.view(tokens_mask.size(0)*tokens_mask.size(1),
                                tokens_mask.size(2)).unsqueeze(2).repeat(1, 1, self.sent_hidden_size)
        bilstm_encoded_tokens = bilstm_encoded_tokens * mask

        bilstm_encoded_tokens = bilstm_encoded_tokens.contiguous().view(batch_size, sent_size, token_size, self.sent_hidden_size)
        masked_bilstm_encoded_tokens = bilstm_encoded_tokens + ((tokens_mask-1)*999).unsqueeze(3).repeat(1, 1, 1, self.sent_hidden_size)
        max_pooled_bilstm_sents = masked_bilstm_encoded_tokens.max(dim=2)[0]  # Batch * sent * dim
        encoded_tokens = bilstm_encoded_tokens

        bilstm_encoded_sents, sent_hidden = self.document_encoder.forward_packed(max_pooled_bilstm_sents, doc_l)
        mask = sent_mask.unsqueeze(2).repeat(1,1, self.doc_hidden_size)
        bilstm_encoded_sents = bilstm_encoded_sents * mask
        encoded_sents = bilstm_encoded_sents.unsqueeze(1).repeat(1, token_size, 1, 1).view(batch_size, sent_size*token_size,
                                                                                           bilstm_encoded_sents.size(2))
        encoded_tokens = encoded_tokens.contiguous().view(batch_size, sent_size*token_size, encoded_tokens.size(3))
        encoded_tokens = torch.cat([tk, encoded_sents], dim=2)
        max_pooled_doc = encoded_tokens.max(dim=1)[0]

        encoder_output = {"encoded_tokens": encoded_tokens,
                          "token_hidden": token_hidden,
                          "encoded_sents": bilstm_encoded_sents,
                          "sent_hidden": sent_hidden,
                          "document_rep" : max_pooled_doc,
                          "token_attention_matrix" : None,
                          "sent_attention_matrix" : None}

        return encoder_output