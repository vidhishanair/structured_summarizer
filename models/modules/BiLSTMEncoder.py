import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np


class BiLSTMEncoder(nn.Module):
    def __init__(self, device, hidden_size, input_size, num_layers, dropout=0.5, bidirectional=True):
        super(BiLSTMEncoder, self).__init__()
        if bidirectional:
            hidden_size = hidden_size//2
        self.device = device
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)

    def forward(self, input, seq_len):
        """
        Returns BiLSTM encoded sequence
        :param input: batch*document_size*sent_size*emb_size
        :return: batch*document_size*sent_size*hidden_dim
        """
        # print(seq_len)
        # pack = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len, batch_first=True)
        output, hidden = self.bilstm(input)
        # output, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)

        return output, hidden

    def forward_packed(self, input, seq_len_tensor):
        # Sort by length (keep idx)
        seq_len = np.array(seq_len_tensor.cpu())
        #sent_len, idx_sort = np.sort(seq_len)[::-1], np.argsort(-seq_len)
        #idx_unsort = np.argsort(idx_sort)

        #idx_sort = input.new_tensor(torch.from_numpy(idx_sort), dtype=torch.long) #.to(self.device)
        #sent_variable = input.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        #print(seq_len)
        #print(sent_len)
        batch_size, total_sequence_length, prev_dim = input.size()
        sent_packed = nn.utils.rnn.pack_padded_sequence(input, seq_len.copy(), batch_first=True, enforce_sorted=False)
        sent_output, hidden = self.bilstm(sent_packed)
        unpacked_sequence_tensor = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]

        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.new_zeros(
                batch_size, sequence_length_difference, unpacked_sequence_tensor.size(-1)
            )
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        # Un-sort by length
        #idx_unsort = input.new_tensor(torch.from_numpy(idx_unsort), dtype=torch.long) #.to(self.device)
        #sent_output = sent_output.index_select(0, idx_unsort)

        #del idx_sort, idx_unsort
        return unpacked_sequence_tensor, hidden
