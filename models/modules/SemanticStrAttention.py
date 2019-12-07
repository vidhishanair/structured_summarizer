import torch.nn as nn
import torch
import torch.nn.functional as F

from models.modules.BilinearMatrixAttention import BilinearMatrixAttention


class SemanticStrAttention(nn.Module):
    def __init__(self, device, sem_dim_size, sent_hiddent_size, bidirectional, py_version):
        super(SemanticStrAttention, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.sem_dim_size = sem_dim_size
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size
        self.pytorch_version = py_version
        print("Setting pytorch "+self.pytorch_version+" version for Structured Attention")

        self.tp_linear = nn.Linear(sent_hiddent_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tp_linear.weight)
        nn.init.constant_(self.tp_linear.bias, 0)


        self.fzlinear = nn.Linear(self.str_dim_size, self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.fzlinear.weight)
        nn.init.constant_(self.fzlinear.bias, 0)

    def forward(self, input, adj_mat): #batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()

        adj_mat = adj_mat + 0.005
        row_sums = torch.sum(adj_mat, dim=2, keepdim=True)
        adj_mat = adj_mat/row_sums.expand_as(adj_mat)
        # all_head = adj_mat.clone()
        # row_sums = torch.sum(adj_mat, dim=2, keepdim=True)
        # all_head[row_sums.expand_as(adj_mat)!=0] = adj_mat[row_sums.expand_as(adj_mat)!=0]/row_sums.expand_as(adj_mat)[row_sums.expand_as(adj_mat)!=0]

        tp = F.tanh(self.tp_linear(input)) # b*s, token, h1
        attended_hidden = torch.bmm(adj_mat, tp)
        output = F.relu(self.fzlinear(attended_hidden))
        #output = F.tanh(self.fzlinear(finp))

        #self.output = output

        return output