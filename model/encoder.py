from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops import roi_pool

from . import resnet


class Encoder(nn.Module):

    def __init__(self,
                 char_embedding_dim: int,
                 out_dim: int,
                 image_feature_dim: int = 512,
                 nheaders: int = 8,
                 nlayers: int = 6,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 image_encoder: str = 'resnet50',
                 roi_pooling_mode: str = 'roi_align',
                 roi_pooling_size: Tuple[int, int] = (7, 7)):
        
        super().__init__()

        self.dropout = dropout
        assert roi_pooling_mode in ['roi_align', 'roi_pool'], 'roi pooling model: {} not support.'.format(
            roi_pooling_mode)
        self.roi_pooling_mode = roi_pooling_mode
        assert roi_pooling_size and len(roi_pooling_size) == 2, 'roi_pooling_size not be set properly.'
        self.roi_pooling_size = tuple(roi_pooling_size) 

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=char_embedding_dim,
                                                               nhead=nheaders,
                                                               dim_feedforward=feedforward_dim,
                                                               dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=nlayers)

        
            
        if image_encoder == 'resnet18':
            self.cnn = resnet.resnet18(output_channels=out_dim)
            
        elif image_encoder == 'resnet34':
            self.cnn = resnet.resnet34(output_channels=out_dim)
            
        elif image_encoder == 'resnet50':
            self.cnn = resnet.resnet50(output_channels=out_dim)
            
        elif image_encoder == 'resnet101':
            self.cnn = resnet.resnet101(output_channels=out_dim)
            
        elif image_encoder == 'resnet152':
            self.cnn = resnet.resnet152(output_channels=out_dim)
        else:
            raise NotImplementedError()
           
        self.conv = nn.Conv2d(image_feature_dim, out_dim, self.roi_pooling_size)
        
        self.bn = nn.BatchNorm2d(out_dim)

        self.projection = nn.Linear(2 * out_dim, out_dim)
        
        self.norm = nn.LayerNorm(out_dim)

        
        position_embedding = torch.zeros(max_len, char_embedding_dim)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, char_embedding_dim, 2).float() *
                             -(math.log(10000.0) / char_embedding_dim))
        
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)
          
        self.register_buffer('position_embedding', position_embedding)

        self.pe_droput = nn.Dropout(self.dropout)

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor, transcripts: torch.Tensor,
                src_key_padding_mask: torch.Tensor):
        

        B, N, T, D = transcripts.shape

        
        _, _, origin_H, origin_W = images.shape

       
        images = self.cnn(images)
        _, C, H, W = images.shape

       
        rois_batch = torch.zeros(B, N, 5, device=images.device)
        for i in range(B):  
           
            doc_boxes = boxes_coordinate[i]
            
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1)
            
            rois_batch[i, :, 1:5] = pos
            
            rois_batch[i, :, 0] = i

        spatial_scale = float(H / origin_H)
       
        if self.roi_pooling_mode == 'roi_align':
            image_segments = roi_align(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
            
        else:
            image_segments = roi_pool(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)

        
        image_segments = F.relu(self.bn(self.conv(image_segments)))
        
        image_segments = image_segments.squeeze()

      
        image_segments = image_segments.unsqueeze(dim=1)

       
        transcripts_segments = self.pe_droput(transcripts + self.position_embedding[:, :, :transcripts.size(2), :])
      
        transcripts_segments = transcripts_segments.reshape(B * N, T, D)

      
        image_segments = image_segments.expand_as(transcripts_segments)

       
        out = image_segments + transcripts_segments

       
        out = out.transpose(0, 1).contiguous()

        
        out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)

        
        out = out.transpose(0, 1).contiguous()
        
        out = self.norm(out)
        
        out = F.dropout(out, p=self.dropout)

        return out
