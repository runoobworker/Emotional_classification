import torch
import math

# 位置编码
class Positional_Encoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(Positional_Encoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0), torch.arange(0, d_model, 2).float() / d_model)
        div_term1 = torch.pow(torch.tensor(10000.0), torch.arange(1, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# transformer模型
class Transformer(torch.nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)  # 词嵌入层
        self.position_embedding = Positional_Encoding(config.d_model, config.dropout, config.max_len)  # 位置编码

        encoder = torch.nn.TransformerEncoderLayer(
            config.d_model,
            config.num_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='relu',
            batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder, config.num_encoder_layer)
        '''
        decoder = torch.nn.TransformerDecoderLayer(config.d_model,
            config.num_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='relu',
            batch_first=True)
        self.transformer = torch.nn.TransformerDecoder(decoder, config.num_encoder_layer)
        '''
        # self.transformer = torch.nn.Transformer(d_model=config.d_model, nhead=config.num_head, num_encoder_layers=config.num_encoder_layer, num_decoder_layers=config.num_decoder_layer, dim_feedforward=config.dim_feedforward, dropout=config.dropout, activation='relu')  # transformer模型
        self.fc = torch.nn.Linear(config.d_model, config.num_class)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        emb_e = self.embedding(x)
        in_e = self.position_embedding(emb_e)
        out = self.transformer(in_e)
        out = out[:, 0, :]
        out = self.fc(out)
        out = self.softmax(out)
        return out
