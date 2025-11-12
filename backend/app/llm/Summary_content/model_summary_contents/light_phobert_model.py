import math

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
import torch


class LightweightPhoBERTSUM(nn.Module):
    def __init__(self, phobert_model_name="vinai/phobert-base", hidden_size=768, dropout=0.1, max_sent=96):
        super().__init__()
        config = AutoConfig.from_pretrained(phobert_model_name)
        config.num_hidden_layers = 6
        
        self.encoder = AutoModel.from_pretrained(phobert_model_name, config=config)
        
        small_hidden_size = 384
        self.dim_reduction = nn.Linear(hidden_size, small_hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=small_hidden_size,
            nhead=6,  # Giảm số lượng attention heads
            dropout=dropout,
            batch_first=True,
            dim_feedforward=1024  # Giảm kích thước feedforward
        )
        
        self.inter_sentence_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )
        
        self.layer_norm = nn.LayerNorm(small_hidden_size)
        self.classifier = nn.Linear(small_hidden_size, 1)
        self.register_buffer("pos_encoder", self.__create_positional_encoding(max_sent, small_hidden_size))
    
    def __create_positional_encoding(self, max_len, d_model):
        """Tạo positional encoding cho các câu"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, input_ids, attention_mask):
        batch_size, num_sent, seq_len = input_ids.size()
        
        # Xử lý từng câu qua PhoBERT
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy vector [CLS] và reshape lại
        cls_vec = outputs.last_hidden_state[:, 0, :]
        cls_vec = cls_vec.view(batch_size, num_sent, -1)  # [batch, num_sent, hidden]
        
        # Giảm chiều
        cls_vec = self.dim_reduction(cls_vec)
        
        # Tạo padding mask
        sent_attention = attention_mask.view(batch_size, num_sent, -1)
        sent_mask = (sent_attention.sum(dim=2) == 0)  # True cho các câu padding
        
        # Thêm positional encoding
        if num_sent <= self.pos_encoder.size(1):
            cls_vec = cls_vec + self.pos_encoder[:, :num_sent, :]
        
        # Truyền qua transformer với padding mask
        trans_out = self.inter_sentence_transformer(
            cls_vec,
            src_key_padding_mask=sent_mask  # mask các câu padding
        )
        
        # Kết nối residual và layer norm
        trans_out = trans_out + cls_vec  # Kết nối residual
        trans_out = self.layer_norm(trans_out)  # Layer normalization
        
        # Phân loại mỗi câu
        logits = self.classifier(trans_out)
        
        return logits.squeeze(-1)  # [batch, num_sent]
