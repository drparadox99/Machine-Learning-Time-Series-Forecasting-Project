import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False
#
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return self.pe[:, :x.size(1)]
#
#
#
# class PositionalEmbedding(nn.Module):
#     def __init__(self, sequence_length, d_model):
#         super(PositionalEmbedding, self).__init__()
#
#         # Creating positional embeddings as learnable parameters
#         self.positional_embeddings = nn.Parameter(torch.randn(sequence_length, d_model))
#
#     def forward(self, inputs):
#         # Add positional embeddings to the input sequence
#         return inputs + self.positional_embeddings


class Encoder(nn.Module):
    def __init__(self, num_heads: int, d_model: int, ff_dim: int, lookback_w: int,num_series:int,forecast_h:int ,encoder_dropout: float):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.lookback_w = lookback_w
        self.ff_dim = ff_dim
        self.encoder_dropout = encoder_dropout

        self.dropout = nn.Dropout(encoder_dropout)
        # FF Layers
        self.fc_1 = nn.Linear(lookback_w, self.ff_dim)
        self.ff_dropout = nn.Dropout(encoder_dropout)
        self.fc_2 = nn.Linear(self.ff_dim,forecast_h)

        self.fc_pca = nn.Linear(lookback_w, d_model)
        self.fc_pc_inv = nn.Linear(d_model, lookback_w)

        # Multi-attention block
        self.multi_attention_block = nn.MultiheadAttention(embed_dim=self.lookback_w, num_heads=num_heads,
                                                           dropout=encoder_dropout)

    def forward(self, inputs): #[B,num_series,lookback_w]

        attention_out, _ = self.multi_attention_block(inputs, inputs, inputs)
        x = self.dropout(attention_out)
        res = x + inputs  # residual connection
        # FF Layers
        x = self.ff_dropout(x)
        #x = self.fc_2(x)
        return x




class Transformers(nn.Module):
    def __init__(self, lookback_w: int,num_series:int ,forecast_h: int, num_encoders: int, num_heads: int,
                 d_model: int, ff_dim: int, ff_layers_hidden_units: list, encoder_dropout: float,
                 ff_layers_dropout: float):
        super(Transformers, self).__init__()
        self.lookback_w = lookback_w
        self.num_series = num_series
        self.forecast_h = forecast_h
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.ff_layers_hidden_units = ff_layers_hidden_units
        self.encoder_dropout = encoder_dropout
        self.ff_layers_dropout = ff_layers_dropout
        self.encoders = nn.ModuleList([Encoder(num_heads=num_heads, d_model=d_model, ff_dim=ff_dim,
                                               lookback_w=lookback_w,num_series=num_series,forecast_h=forecast_h,encoder_dropout=encoder_dropout) for _ in range(num_encoders)])
        #self.positional_embedding = PositionalEmbedding(num_series, lookback_w)
        self.dropout = nn.Dropout(encoder_dropout)


    def forward(self, x):
        #x = inputs
        seq_last = x[:, -1:, :].detach()  # [B,1,num_series]  #
        x = x - seq_last
        x = x.permute(0,2,1)  #[B,num_series,lookback_w]
        #x = self.positional_embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.permute(0,2,1) #[B,forecast_h,num_series]
        x = x + seq_last
        return x


