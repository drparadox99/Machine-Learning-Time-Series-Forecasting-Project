
import numpy as np
import torch
from torch import nn, optim
from Utils.RevIN import RevIN

from Models.Formers.Layers.BEATS_CELL_Transformer_EncDec import Decoder, DecoderLayer
from Models.Formers.Layers.SelfAttention_Family import FullAttention, AttentionLayer,ProbAttention
from Models.Formers.Embed import DataEmbedding_wo_temp

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class BEATS_CELL_Trans_Dec(nn.Module):

    def __init__(
            self,
            args,
            lookback_w,
            forecast_h,
            num_series,
            hidden_units,
            num_cells,
            cell_dropout,
            dropout,
    ):
        super(BEATS_CELL_Trans_Dec, self).__init__()
        self.forecast_h = forecast_h
        self.lookback_w = lookback_w
        self.num_series = num_series
        self.hidden_units = hidden_units
        self.num_cells = num_cells
        self.hidden_expansion = nn.Linear(lookback_w, hidden_units)
        self.expansion_f = nn.Linear(hidden_units, forecast_h)
        self.cells = nn.ModuleList([BEAT_Block(args,hidden_units,forecast_h,num_series,hidden_units,cell_dropout) for prob in range(num_cells)])
        self.dropout = nn.Dropout(dropout)
        self.revin_layer = RevIN(self.num_series)

    def forward(self, x_enc,x_dec): #[B,lookback_w,num_series]
        x = x_enc
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        #x = self.revin_layer(x, 'norm')
        res = x
        x_s = self.hidden_expansion(x.permute(0,2,1)).permute(0,2,1) #[B,num_series,hidden_units,num_series]  #x_s = x
        x_l = x.permute(0,2,1)#[B,num_series,lookback_w]
        for cell in self.cells:
            x_s, x_l = cell(x_s,x_dec)  #x:[B,hidden_units,num_series] #x_:[B,num_series,hidden_units]
            #x_s = x_s + res  ##res:[B,hidden_units,num_series]
            #x_expansion = self.dropout(x_expansion)

        #forecast = self.expansion_f(x_s.permute(0,2,1)) #+ self.x_f(x)  #[B,num_series,forecast_h]
        #forecast = forecast.permute(0, 2, 1)  # [B,forecast_h,num_series]
        #forecast = self.revin_layer(forecast, 'denorm')

        forecast = x_s + seq_last
        return forecast


class BEAT_Block(nn.Module):

    def __init__(self,args,lookback_w,forecast_h,num_series,hidden_units,cell_dropout, theta_t=4,theta_p=8,device="cpu" ):
        super(BEAT_Block, self).__init__()
        self.device = device

        self.x_linear = nn.Linear(lookback_w,lookback_w )
        self.kernel_size = 25
        self.decomposition = series_decomp(self.kernel_size)

        self.cell_dropout =nn.Dropout(0.9)

        self.process_t = nn.Linear(hidden_units,hidden_units)
        self.process_p = nn.Linear(lookback_w,lookback_w)
        self.mha = nn.MultiheadAttention(embed_dim=lookback_w, num_heads=4,
                                                           dropout=0.0)
        self.forecast_h = forecast_h
        self.sgu = SpatialGatingUnit(lookback_w,num_series)
        self.dec_transformer = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, args.num_series, bias=True)
        )

        self.dec_embedding = DataEmbedding_wo_temp(args.num_series, args.d_model, args.embed, args.freq,
                                               args.dropout)

        self.ln = nn.Linear(args.num_series,forecast_h)
        self.ln_2 = nn.Linear(args.d_model, forecast_h)

    def forward(self, x_expansion,x_dec): #[B,hidden_units,num_series]
        #x_f = self.x_linear(x)
        seasonal_init, trend_init = self.decomposition(x_expansion)  #[B,lookback_w,num_series]
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) #[B,num_series,hidden_dim]
        mha_seasonal = self.cell_dropout(self.process_p(self.mha(seasonal_init,seasonal_init,seasonal_init)[0])).permute(0,2,1) ##[B,lookback_w,num_series]
        #trend_init = self.sgu(trend_init)

        trend_init = self.ln(trend_init.permute(0,2,1)).permute(0,2,1)
        #trend_init = torch.rand(32,96,128)
        x_dec = self.dec_embedding(x_dec)
        dec_out = self.dec_transformer(x_dec,trend_init)[:, -self.forecast_h:, :]  # [B, L, D]

        dec_out = dec_out + self.ln_2(mha_seasonal.permute(0,2,1)).permute(0,2,1)
        dec_out = self.cell_dropout(dec_out)
        #print("dec out", dec_out.shape)
        #print("mha_seasonal",mha_seasonal.shape)
        expansion_f = dec_out #mha_seasonal + dec_out.permute(0,2,1) #trend_init.permute(0,2,1)
        return expansion_f, 0 #x_f #expansion_f[B,num_series,hidden_units]



class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_units, num_series):
        super().__init__()
        self.proj = nn.Linear(hidden_units//2, hidden_units//2)
        self.proj_v = nn.Linear(hidden_units//2,hidden_units)
    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.proj(v)
        output = u * v
        output = self.proj_v(output)
        return output

#
# # selected_model = BEATS_CELL(lookback_w=args.past_history,forecast_h=args.forecast_horizon,hidden_units=128,num_cells=1,cell_dropout=0.0,dropout=0.0)
# #0.7984 BEATS_CELL
# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
#
#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x
#
#
# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
#
#
# class BEATS_CELL(nn.Module):
#
#     def __init__(
#             self,
#             lookback_w,
#             forecast_h,
#             hidden_units,
#             num_cells,
#             cell_dropout,
#             dropout,
#     ):
#         super(BEATS_CELL, self).__init__()
#         self.forecast_h = forecast_h
#         self.lookback_w = lookback_w
#         self.hidden_units = hidden_units
#         self.num_cells = num_cells
#         self.hidden_expansion = nn.Linear(lookback_w, hidden_units)
#         self.expansion_f = nn.Linear(hidden_units, forecast_h)
#         self.x_f = nn.Linear(lookback_w, forecast_h)
#         self.cells = nn.ModuleList([BEAT_Block(lookback_w,forecast_h,hidden_units,cell_dropout) for prob in range(num_cells)])
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x): #[B,num_series,lookback_w]
#         seq_last = x[:, -1:, :].detach()
#         x = x - seq_last
#         x = x.permute(0,2,1)  #[B,num_series,lookback_w]
#         x_expansion = self.hidden_expansion(x).permute(0,2,1) #[B,hidden_units,num_series]
#         res = x
#         for cell in self.cells:
#             x_, x = cell(x_expansion,x)  #expansion_f:[B,num_series,hidden_units] #x:[B,num_series,forecast_h]
#             #x_ = x_ + res
#             #x_expansion = self.dropout(x_expansion)
#
#         forecast = self.expansion_f(x_) #+ self.x_f(x)  #[B,num_series,forecast_h]
#         forecast = forecast.permute(0, 2, 1)  # [B,forecast_h,num_series]
#         forecast = forecast + seq_last
#         return forecast
#
#
# class BEAT_Block(nn.Module):
#
#     def __init__(self,lookback_w,forecast_h,hidden_units,cell_dropout, theta_t=4,theta_p=8,device="cpu" ):
#         super(BEAT_Block, self).__init__()
#         self.device = device
#
#         self.x_linear = nn.Linear(lookback_w,lookback_w )
#         self.kernel_size = 25
#         self.decomposition = series_decomp(self.kernel_size)
#
#         self.cell_dropout =nn.Dropout(0.2)
#
#         self.process_t = nn.Linear(hidden_units,hidden_units)
#         self.process_p = nn.Linear(hidden_units,hidden_units)
#         self.mha = nn.MultiheadAttention(embed_dim=hidden_units, num_heads=4,
#                                                            dropout=0.2)
#
#     def forward(self, x_expansion,x): #[B,num_series,hidden_units]
#         x_f = self.x_linear(x)
#         seasonal_init, trend_init = self.decomposition(x_expansion)  #[B,hidden_units,num_series]
#         seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) #[B,num_series,hidden_dim]
#         #mha_seasonal = self.cell_dropout(self.process_p(self.mha(seasonal_init,seasonal_init,seasonal_init)[0]))
#         mha_seasonal = self.cell_dropout(self.process_p(self.mha(seasonal_init,seasonal_init,seasonal_init)[0]))
#
#         #trend_init = self.cell_dropout(self.process_t(trend_init))
#
#         expansion_f = mha_seasonal #+ trend_init
#         return expansion_f, x_f #expansion_f[B,num_series,hidden_units]
#
