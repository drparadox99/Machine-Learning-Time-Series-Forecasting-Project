from torch import nn
from torch.nn import functional as F
import torch
class SpatialGatingUnit(nn.Module):
    def __init__(self,input_dim, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)  # normalizes last dim
        self.spatial_proj = nn.Conv1d(input_dim[1], input_dim[1], kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        #[B,num_series,d_ffn*2]
        u, v = x.chunk(2, dim=-1) #u & v [B,num_series,d_ffn]
        #v = self.norm(v)
        v = self.spatial_proj(v) #[B,num_series,d_ffn]
        out = u * v  #B,nuum_series,dim]
        return out


class gMLPBlock(nn.Module):
    def __init__(self,input_dim, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim[0])
        self.channel_proj1 = nn.Linear(input_dim[0], d_ffn*2)
        self.sgu = SpatialGatingUnit(input_dim,d_ffn)
        self.channel_proj2 = nn.Linear(d_ffn, input_dim[0])
        self.dropout = nn.Dropout(0.9)
        self.activation = nn.GELU()


    def forward(self, x):
        residual = x     # [B,num_series,seq_len]
        #x = self.norm(x)  # [B,num_series,seq_len]
        x = self.channel_proj1(x)  #[B,num_series,dim_*2]
        #x = self.activation(x)
        x= self.dropout(x)
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        x= self.dropout(x)

        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, d_model):
        super(PositionalEmbedding, self).__init__()
        # Define positional embeddings as a model parameter
        # Initialize using a normal distribution with mean=0.0 and std=1.0
        self.positional_embeddings = nn.Parameter(torch.randn(sequence_length, d_model))

    def forward(self, inputs):
        # Add positional embeddings to the input sequence
        return inputs + self.positional_embeddings
class GMLP(nn.Module):
    def __init__( self,input_dim,forecast_horizon, d_ffn=128, num_layers=1):
        super().__init__()
        self.blocks = nn.Sequential(*[gMLPBlock(input_dim, d_ffn) for _ in range(num_layers)])
        #self.embed = nn.Embedding(num_tokens, d_model)
        self.output_proj = nn.Linear(input_dim[0], forecast_horizon)
        self.positional_encoding = PositionalEmbedding(input_dim[1],input_dim[0])
        self.dropout = nn.Dropout(0.9)

    def forward(self, x,x_dec=None):
        seq_last = x[:, -1:, :].detach() # [B,1,num_series]  #
        x = x - seq_last
        x = x.permute(0,2,1) #[B,num_series,lookback_w]
        #x = self.positional_encoding(x)
        #embedding = self.embed(x)  # [B,seq_len,d_model]
        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)
        out = self.output_proj(x).permute(0,2,1)
        out = self.dropout(out)
        out = out + seq_last
        return out




#
# class SpatialGatingUnit(nn.Module):
#     def __init__(self,input_dim, d_ffn):
#         super().__init__()
#         self.norm = nn.LayerNorm(d_ffn)  # normalizes last dim
#         self.spatial_proj = nn.Conv1d(input_dim[1], input_dim[1], kernel_size=1)
#         nn.init.constant_(self.spatial_proj.bias, 1.0)
#
#     def forward(self, x):
#         #[B,num_series,d_ffn*2]
#         u, v = x.chunk(2, dim=-1) #u & v [B,num_series,d_ffn]
#         #v = self.norm(v)
#         v = self.spatial_proj(v) #[B,num_series,d_ffn]
#         out = u * v  #B,nuum_series,dim]
#         return out
#
#
# class gMLPBlock(nn.Module):
#     def __init__(self,input_dim, d_ffn):
#         super().__init__()
#         self.norm = nn.LayerNorm(input_dim[0])
#         self.channel_proj1 = nn.Linear(input_dim[0], d_ffn*2)
#         self.channel_proj2 = nn.Linear(d_ffn, input_dim[0])
#         self.sgu = SpatialGatingUnit(input_dim,d_ffn)
#         self.dropout = nn.Dropout(0.9)
#         self.activation = nn.GELU()
#
#
#     def forward(self, x):
#         residual = x     # [B,num_series,seq_len]
#         #x = self.norm(x)  # [B,num_series,seq_len]
#         x = self.channel_proj1(x)  #[B,num_series,dim_*2]
#         #x = self.activation(x)
#         x= self.dropout(x)
#         x = self.sgu(x)
#         x = self.channel_proj2(x)
#         out = x + residual
#         x= self.dropout(x)
#
#         return out
#
# class PositionalEmbedding(nn.Module):
#     def __init__(self, sequence_length, d_model):
#         super(PositionalEmbedding, self).__init__()
#         # Define positional embeddings as a model parameter
#         # Initialize using a normal distribution with mean=0.0 and std=1.0
#         self.positional_embeddings = nn.Parameter(torch.randn(sequence_length, d_model))
#
#     def forward(self, inputs):
#         # Add positional embeddings to the input sequence
#         return inputs + self.positional_embeddings
# class GMLP(nn.Module):
#     def __init__( self,input_dim,forecast_horizon, d_ffn=128, num_layers=2):
#         super().__init__()
#         self.blocks = nn.Sequential(*[gMLPBlock(input_dim, d_ffn) for _ in range(num_layers)])
#         #self.embed = nn.Embedding(num_tokens, d_model)
#         self.output_proj = nn.Linear(input_dim[0], forecast_horizon)
#         self.positional_encoding = PositionalEmbedding(input_dim[1],input_dim[0])
#         self.dropout = nn.Dropout(0.9)
#
#     def forward(self, x):
#         seq_last = x[:, -1:, :].detach() # [B,1,num_series]  #
#         x = x - seq_last
#         x = x.permute(0,2,1) #[B,num_series,lookback_w]
#         #x = self.positional_encoding(x)
#         #embedding = self.embed(x)  # [B,seq_len,d_model]
#         for block in self.blocks:
#             x = block(x)
#             x = self.dropout(x)
#         out = self.output_proj(x).permute(0,2,1)
#         out = out + seq_last
#         return out
#

