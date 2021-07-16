# @Time   : 2021/7/16
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

r"""
GCE-GNN
################################################

Reference:
    Ziyang Wang et al. "Global Context Enhanced Graph Neural Networks for Session-based Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/CCIIPLab/GCE-GNN

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class LocalAggregator(MessagePassing):
    def __init__(self, dim, alpha):
        super().__init__(aggr='add')
        self.edge_emb = nn.Embedding(4, dim)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        x = x_j * x_i
        a = self.edge_emb(edge_attr)
        e = (x * a).sum(dim=-1)
        e = self.leakyrelu(e)
        e = softmax(e, index, ptr, size_i)
        return e.unsqueeze(-1) * x_j


class GCEGNN(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GCEGNN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.leakyrelu_alpha = config['leakyrelu_alpha']
        self.dropout_local = config['dropout_local']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.max_seq_length = dataset.field2seqlen[self.ITEM_SEQ]

        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)

        # define layers and loss
        self.local_agg = LocalAggregator(self.embedding_size, self.leakyrelu_alpha)
        self.w_1 = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False)
        self.w_2 = nn.Linear(self.embedding_size, 1, bias=False)
        self.glu1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.glu2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def fusion(self, hidden, mask):
        batch_size = hidden.shape[0]
        length = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:length]
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).expand(-1, length, -1)
        nh = self.w_1(torch.cat([pos_emb, hidden], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = self.w_2(nh)
        beta = beta * mask
        final_h = torch.sum(beta * hidden, 1)
        return final_h

    def forward(self, x, edge_index, edge_attr, alias_inputs, item_seq_len):
        mask = alias_inputs.gt(0).unsqueeze(-1)
        h = self.item_embedding(x)
        h_local = self.local_agg(h, edge_index, edge_attr)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_local = h_local[alias_inputs]
        h_local = self.fusion(h_local, mask)
        return h_local

    def calculate_loss(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        edge_attr = interaction['edge_attr']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, edge_attr, alias_inputs, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        test_item = interaction[self.ITEM_ID]
        x = interaction['x']
        edge_index = interaction['edge_index']
        edge_attr = interaction['edge_attr']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, edge_attr, alias_inputs, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        x = interaction['x']
        edge_index = interaction['edge_index']
        edge_attr = interaction['edge_attr']
        alias_inputs = interaction['alias_inputs']
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(x, edge_index, edge_attr, alias_inputs, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
