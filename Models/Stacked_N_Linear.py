
from torch import nn
from torch.nn import functional as F
import torch
from torch import einsum



class Stack_N_Linear(nn.Module):
    def __init__(self, lookback_w, forecast_horizon,num_series,num_blocks,d_ffn,dropout):
        super(Stack_N_Linear,self).__init__()
        self.lookback_w = lookback_w
        self.forecast_horizon = forecast_horizon
        self.blocks = nn.ModuleList(
            [N_Linear_Block(self.lookback_w,d_ffn,dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.d_ffn = d_ffn
        self.output_l = nn.Linear(self.lookback_w, forecast_horizon)

        #self.sgu = SpatialGatingUnit(lookback_w,num_series)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach() # [B,lookback_w,num_series]  #
        x = x - seq_last
        x = x.permute(0, 2, 1)  #[B,num_series,lookback_w]
        for block in self.blocks:
            #x = self.sgu(x)
            x = block(x)
            #x = self.dropout(x)
        x = self.output_l(x)
        x = x.permute(0, 2, 1)  #[B,lookback_w,num_series]
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


class N_Linear_Block(nn.Module):
    def __init__(self,lookback_w,d_ffn,dropout):
        super(N_Linear_Block, self).__init__()

        self.linear = nn.Linear(lookback_w, lookback_w)
        self.dropout = nn.Dropout(dropout)
        self.sgu = SpatialGatingUnit(lookback_w)
        self.decomposition = series_decomp(25)

    def forward(self, x):     #x:[B,lookback_w,num_series]
        #seasonal_init, trend_init = self.decomposition(x.permute(0,2,1))  # [B,lookback_w,num_series]
        #seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2,1)  # [B,num_series,hidden_dim]
        #x =  self.sgu(x) #+ self.sgu(trend_init) #+self.linear(x)


        return x

    # q, k, v = qkv.chunk(3, dim=-1)

class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.proj = nn.Linear(hidden_units//2, hidden_units//2)
        self.proj_v = nn.Linear(hidden_units//2,hidden_units)
    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.proj(v)
        output = u * v
        output = self.proj_v(output)
        return output


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
