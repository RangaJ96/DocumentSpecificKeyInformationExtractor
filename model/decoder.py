from typing import *
import logging

import torch
import torch.nn as nn
from torch import Tensor

from .crf import ConditionalRandomField
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls
from data_utils import documents


class MLPLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None,
                 layer_norm: bool = False,
                 dropout: Optional[float] = 0.0,
                 activation: Optional[str] = 'relu'):

        super().__init__()
        layers = []
        activation_layer = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU
        }

        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(activation_layer.get(activation, nn.Identity()))
                logger.warning(
                    'Rreplace with Identity layer.'.format(activation))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout:
                    layers.append(nn.Dropout(dropout))
                in_dim = dim

        if not out_dim:
            layers.append(nn.Identity())
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim if out_dim else hidden_dims[-1]

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(input, 1))


class BiLSTMLayer(nn.Module):

    def __init__(self, lstm_kwargs, mlp_kwargs):
        super().__init__()
        self.lstm = nn.LSTM(**lstm_kwargs)
        self.mlp = MLPLayer(**mlp_kwargs)

    @staticmethod
    def sort_tensor(x: torch.Tensor, length: torch.Tensor, h_0: torch.Tensor = None, c_0: torch.Tensor = None):
        sorted_lenght, sorted_order = torch.sort(length, descending=True)
        _, invert_order = sorted_order.sort(0, descending=False)
        if h_0 is not None:
            h_0 = h_0[:, sorted_order, :]
        if c_0 is not None:
            c_0 = c_0[:, sorted_order, :]
        return x[sorted_order], sorted_lenght, invert_order, h_0, c_0

    def forward(self, x_seq: torch.Tensor,
                lenghts: torch.Tensor,
                initial: Tuple[torch.Tensor, torch.Tensor]):

        x_seq, sorted_lengths, invert_order, h_0, c_0 = self.sort_tensor(x_seq, lenghts, initial[0], initial[0])
        packed_x = nn.utils.rnn.pack_padded_sequence(x_seq, lengths=sorted_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
                                                     padding_value=keys_vocab_cls.stoi['<pad>'])

        output = output[invert_order]
        logits = self.mlp(output)

        return logits


class UnionLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, x_gcn: Tensor, mask: Tensor, length: Tensor, tags):

        B, N, T, D = x.shape
        x = x.reshape(B, N * T, -1)
        mask = mask.reshape(B, N * T)

        doc_seq_len = length.sum(dim=-1)

        max_doc_seq_len = doc_seq_len.max()

        new_x = torch.zeros_like(x, device=x.device)

        new_mask = torch.zeros_like(mask, device=x.device)

        if self.training:

            tags = tags.reshape(B, N * T)
            
            new_tag = torch.full_like(tags, iob_labels_vocab_cls.stoi['<pad>'], device=x.device)
            
            new_tag = new_tag[:, :max_doc_seq_len]

        for i in range(B):
            doc_x = x[i]
            
            doc_mask = mask[i]
            
            valid_doc_x = doc_x[doc_mask == 1]
            
            num_valid = valid_doc_x.size(0)
            
            new_x[i, :num_valid] = valid_doc_x
            
            new_mask[i, :doc_seq_len[i]] = 1

            if self.training:
                
                valid_tag = tags[i][doc_mask == 1]
                
                new_tag[i, :num_valid] = valid_tag

        new_x = new_x[:, :max_doc_seq_len, :]

        new_mask = new_mask[:, :max_doc_seq_len]

        x_gcn = x_gcn.unsqueeze(2).expand(B, N, T, -1)

        x_gcn = x_gcn.reshape(B, N * T, -1)[:, :max_doc_seq_len, :]

        new_x = x_gcn + new_x

        if self.training:
            return new_x, new_mask, doc_seq_len, new_tag
        else:
            return new_x, new_mask, doc_seq_len, None


class Decoder(nn.Module):

    def __init__(self, bilstm_kwargs, mlp_kwargs, crf_kwargs):
        
        super().__init__()
        
        self.union_layer = UnionLayer()
        
        self.bilstm_layer = BiLSTMLayer(bilstm_kwargs, mlp_kwargs)
        
        self.crf_layer = ConditionalRandomField(**crf_kwargs)

    def forward(self, x: Tensor, x_gcn: Tensor, mask: Tensor, length: Tensor, tags: Tensor):

        new_x, new_mask, doc_seq_len, new_tag = self.union_layer(x, x_gcn, mask, length, tags)

        logits = self.bilstm_layer(new_x, doc_seq_len, (None, None))

        log_likelihood = None
        if self.training:

            log_likelihood = self.crf_layer(logits,
                                            new_tag,
                                            mask=new_mask,
                                            input_batch_first=True,
                                            keepdim=True)

        return logits, new_mask, log_likelihood
