from typing import *
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from data_utils import documents


class GraphLearningLayer(nn.Module):
    def __init__(self, in_dim: int, learning_dim: int, gamma: float, eta: float):
        super().__init__()

        self.projection = nn.Linear(in_dim, learning_dim, bias=False)

        self.learn_w = nn.Parameter(torch.empty(learning_dim))

        self.gamma = gamma

        self.eta = eta

        self.inint_parameters()

    def inint_parameters(self):
        nn.init.uniform_(self.learn_w, a=0, b=1)

    def forward(self, x: Tensor, adj: Tensor, box_num: Tensor = None):

        B, N, D = x.shape

        x_hat = self.projection(x)
        _, _, learning_dim = x_hat.shape

        x_i = x_hat.unsqueeze(2).expand(B, N, N, learning_dim)

        x_j = x_hat.unsqueeze(1).expand(B, N, N, learning_dim)

        distance = torch.abs(x_i - x_j)

        if box_num is not None:

            mask = self.compute_dynamic_mask(box_num)

            distance = distance + mask

        distance = torch.einsum('bijd, d->bij', distance, self.learn_w)
        out = F.leaky_relu(distance)

        max_out_v, _ = out.max(dim=-1, keepdim=True)
        out = out - max_out_v

        soft_adj = torch.exp(out)
        soft_adj = adj * soft_adj

        sum_out = soft_adj.sum(dim=-1, keepdim=True)
        soft_adj = soft_adj / sum_out + 1e-10

        gl_loss = None
        if self.training:
            gl_loss = self._graph_learning_loss(x_hat, soft_adj, box_num)

        return soft_adj, gl_loss

    def compute_static_mask(box_num: Tensor):

        max_len = documents.MAX_BOXES_NUM

        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))

        box_num = box_num.expand_as(mask)

        mask = mask < box_num

        row_mask = mask.unsqueeze(1)

        column_mask = mask.unsqueeze(2)

        mask = (row_mask & column_mask)

        mask = ~mask * -1

        return mask.unsqueeze(-1)

    def compute_dynamic_mask(box_num: Tensor):

        max_len = torch.max(box_num)

        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))

        box_num = box_num.expand_as(mask)
        mask = mask < box_num

        row_mask = mask.unsqueeze(1)

        column_mask = mask.unsqueeze(2)

        mask = (row_mask & column_mask)

        mask = ~mask * -1

        return mask.unsqueeze(-1)

    def _graph_learning_loss(self, x_hat: Tensor, adj: Tensor, box_num: Tensor):

        B, N, D = x_hat.shape

        x_i = x_hat.unsqueeze(2).expand(B, N, N, D)

        x_j = x_hat.unsqueeze(1).expand(B, N, N, D)

        box_num_div = 1 / torch.pow(box_num.float(), 2)

        dist_loss = adj + self.eta * torch.norm(x_i - x_j, dim=3)

        dist_loss = torch.exp(dist_loss)

        dist_loss = torch.sum(dist_loss, dim=(1, 2)) * box_num_div.squeeze(-1)

        f_norm = torch.norm(adj, dim=(1, 2))

        gl_loss = dist_loss + self.gamma * f_norm

        return gl_loss


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):

        super().__init__()

        self.w_alpha = nn.Parameter(torch.empty(in_dim, out_dim))

        self.w_vi = nn.Parameter(torch.empty(in_dim, in_dim))

        self.w_vj = nn.Parameter(torch.empty(in_dim, in_dim))

        self.bias_h = nn.Parameter(torch.empty(in_dim))

        self.w_node = nn.Parameter(torch.empty(in_dim, out_dim))

        self.inint_parameters()

    def inint_parameters(self):

        nn.init.kaiming_uniform_(self.w_alpha, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.w_vi, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.w_vj, a=math.sqrt(5))

        nn.init.uniform_(self.bias_h, a=0, b=1)

        nn.init.kaiming_uniform_(self.w_node, a=math.sqrt(5))

    def forward(self, x: Tensor, alpha: Tensor, adj: Tensor, box_num: Tensor):

        B, N, in_dim = x.shape

        x_i = x.unsqueeze(2).expand(B, N, N, in_dim)

        x_j = x.unsqueeze(1).expand(B, N, N, in_dim)

        x_i = torch.einsum('bijd, dk->bijk', x_i, self.w_vi)

        x_j = torch.einsum('bijd, dk->bijk', x_j, self.w_vj)

        H = F.relu(x_i + x_j + alpha + self.bias_h)

        AH = torch.einsum('bij, bijd-> bid', adj, H)

        new_x = torch.einsum('bid,dk->bik', AH, self.w_node)

        new_x = F.relu(new_x)

        new_alpha = torch.einsum('bijd,dk->bijk', H, self.w_alpha)

        new_alpha = F.relu(new_alpha)

        return new_x, new_alpha


class GLCN(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 gamma: float = 0.0001,
                 eta: float = 1,
                 learning_dim: int = 128,
                 num_layers=2):

        super().__init__()

        self.gl_layer = GraphLearningLayer(in_dim=in_dim, gamma=gamma, eta=eta, learning_dim=learning_dim)
        modules = []
        in_dim_cur = in_dim
        for i in range(num_layers):

            m = GCNLayer(in_dim_cur, out_dim)

            in_dim_cur = out_dim

            out_dim = in_dim_cur

            modules.append(m)

        self.gcn = nn.ModuleList(modules)

        self.alpha_transform = nn.Linear(6, in_dim, bias=False)

    def forward(self, x: Tensor, rel_features: Tensor, adj: Tensor, box_num: Tensor, **kwargs):

        alpha = self.alpha_transform(rel_features)

        soft_adj, gl_loss = self.gl_layer(x, adj, box_num)

        adj = adj * soft_adj

        for i, gcn_layer in enumerate(self.gcn):
            x, alpha = gcn_layer(x, alpha, adj, box_num)

        return x, soft_adj, gl_loss
