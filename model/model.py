from typing import *

import torch
import torch.nn as nn
import numpy as np

from .encoder import Encoder
from .graph import GLCN
from .decoder import Decoder
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls


class Model(nn.Module):

    def __init__(self, **arg):
        super().__init__()
        
        embedding= arg['embedding']
        
        encoder = arg['encoder']
        
        graph = arg['graph']
        
        decoder = arg['decoder']
        
        self.make_model(embedding, encoder, graph, decoder)

    def make_model(self, embedding, encoder, graph, decoder):

        embedding['num_embeddings'] = len(keys_vocab_cls)
        
        self.word_emb = nn.Embedding(**embedding)

        encoder['char_embedding_dim'] = embedding['embedding_dim']
        
        self.encoder = Encoder(**encoder)

        graph['in_dim'] = encoder['out_dim']
        
        graph['out_dim'] = encoder['out_dim']
        
        self.graph = GLCN(**graph)

        decoder['bilstm']['input_size'] = encoder['out_dim']
        
        if decoder['bilstm']['bidirectional']:
            decoder['mlp']['in_dim'] = decoder['bilstm']['hidden_size'] * 2
        else:
            decoder['mlp']['in_dim'] = decoder['bilstm']['hidden_size']
            
        decoder['mlp']['out_dim'] = len(iob_labels_vocab_cls)
        decoder['crf']['num_tags'] = len(iob_labels_vocab_cls)
        self.decoder = Decoder(**decoder)

    def _pooling(self, input, text_mask):
        
        input = input * text_mask.detach().unsqueeze(2).float()
        
        sum_out = torch.sum(input, dim=1)
        
        text_len = text_mask.float().sum(dim=1)
        
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
		
        text_len = text_len + text_len.eq(0).float()  
        
        mean_out = sum_out.div(text_len)
        return mean_out

    
    def compute_mask(mask: torch.Tensor):
        
        B, N, T = mask.shape
		
        mask = mask.reshape(B * N, T)
		
        mask_sum = mask.sum(dim=-1)  


        graph_node_mask = mask_sum != 0
       
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T) 
        
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  
		
        return src_key_padding_mask, graph_node_mask

    def forward(self, **arg):
      
        whole_image = arg['whole_image']  
		
        relation_features = arg['relation_features']  
		
        text_segments = arg['text_segments']  
		
        text_length = arg['text_length'] 
		
        iob_tags_label = arg['iob_tags_label'] if self.training else None 
		
        mask = arg['mask']  
        boxes_coordinate = arg['boxes_coordinate']  

       
        text_emb = self.word_emb(text_segments)

        
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)

        
        x = self.encoder(images=whole_image, boxes_coordinate=boxes_coordinate, transcripts=text_emb,
                         src_key_padding_mask=src_key_padding_mask)

       
        text_mask = torch.logical_not(src_key_padding_mask).byte()
     
        x_gcn = self._aggregate_avg_pooling(x, text_mask)
       
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
   
        x_gcn = x_gcn * graph_node_mask.byte()

       
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)  
       
        x_gcn = x_gcn.reshape(B, N, -1)
       
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj

      
        logits, new_mask, log_likelihood = self.decoder(x.reshape(B, N, T, -1), x_gcn, mask, text_length,
                                                        iob_tags_label)
       

        output = {"logits": logits, "new_mask": new_mask, "adj": adj}

        if self.training:
            output['gl_loss'] = gl_loss
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
        return output

    def __strd__(self):
       
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
       
