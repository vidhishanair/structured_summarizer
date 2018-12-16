from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn

from models.modules.BiLSTMEncoder import BiLSTMEncoder
from models.modules.StructuredAttention import StructuredAttention
from utils import config
from numpy import random
import itertools

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class StructuredEncoder(nn.Module):
    def __init__(self):
        super(StructuredEncoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.drop = nn.Dropout(0.3)
        init_wt_normal(self.embedding.weight)
        bidirectional = True
        device = torch.device("cuda" if config.use_gpu else "cpu")

        #self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        if bidirectional:
            self.sem_dim_size = 2*config.sem_dim_size
            self.sent_hidden_size = 2*config.hidden_dim
            self.doc_hidden_size = 2*config.hidden_dim
        else:
            self.sem_dim_size = config.sem_dim_size
            self.sent_hidden_size = config.hidden_dim
            self.doc_hidden_size = config.hidden_dim

        self.sentence_encoder = BiLSTMEncoder(device, self.sent_hidden_size, config.emb_dim, 1, dropout=0.3, bidirectional=bidirectional)
        self.document_encoder = BiLSTMEncoder(device, self.doc_hidden_size, self.sem_dim_size, 1, dropout=0.3, bidirectional=bidirectional)

        self.sentence_structure_att = StructuredAttention(device, self.sem_dim_size, self.sent_hidden_size, bidirectional, "nightly")
        self.document_structure_att = StructuredAttention(device, self.sem_dim_size, self.doc_hidden_size, bidirectional, "nightly")
        #init_lstm_wt(self.sentence_encoder)
        #init_lstm_wt()

    #seq_lens should be in descending order
    def forward(self, input, sent_l, doc_l, mask_tokens, mask_sents):
        # embedded = self.embedding(input)
        #
        # packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        # output, hidden = self.lstm(packed)
        #
        # h, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        # h = h.contiguous()
        # max_h, _ = h.max(dim=1)

        batch_size, sent_size, token_size = input.size()

        tokens_mask = mask_tokens
        sent_mask = mask_sents

        input = self.embedding(input)
        input = self.drop(input)

        #reshape to 3D tensor
        input = input.contiguous().view(input.size(0)*input.size(1), input.size(2), input.size(3))
        sent_l = list(itertools.chain.from_iterable(sent_l))


        #BiLSTM
        encoded_sentences, hidden = self.sentence_encoder.forward_packed(input, sent_l)

        mask = tokens_mask.view(tokens_mask.size(0)*tokens_mask.size(1), tokens_mask.size(2)).unsqueeze(2).repeat(1,1,encoded_sentences.size(2))
        encoded_sentences = encoded_sentences * mask

        #Structure ATT
        encoded_sentences, sent_attention_matrix = self.sentence_structure_att.forward(encoded_sentences)

        #Reshape and max pool
        encoded_sentences = encoded_sentences.contiguous().view(batch_size, sent_size, token_size, encoded_sentences.size(2))
        encoded_sentences = encoded_sentences + ((tokens_mask-1)*999).unsqueeze(3).repeat(1,1,1,encoded_sentences.size(3))
        encoded_sentences = encoded_sentences.max(dim=2)[0] # Batch * sent * dim

        #Doc BiLSTM
        encoded_documents, hidden = self.document_encoder.forward(encoded_sentences, doc_l)
        mask = sent_mask.unsqueeze(2).repeat(1,1,encoded_documents.size(2))
        encoded_documents = encoded_documents * mask

        #structure Att
        encoded_documents, doc_attention_matrix = self.document_structure_att.forward(encoded_documents)

        #Max Pool
        encoded_documents = encoded_documents + ((sent_mask-1)*999).unsqueeze(2).repeat(1,1,encoded_documents.size(2))
        max_encoded_documents = encoded_documents.max(dim=1)[0]

        return encoded_documents, hidden, max_encoded_documents