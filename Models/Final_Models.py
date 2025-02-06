import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math


def show(_sting, content):
    print(_sting + str(content))

class N_Linear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, input_size, forecast_horizon, individual):
        super(N_Linear, self).__init__()
        self.seq_len = input_size[0]
        self.pred_len = forecast_horizon

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = input_size[1]
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x,number):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, number, :].detach().unsqueeze(1)
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    # def forward(self, x):
    #     # padding on the both ends of time series
    #     front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    #     end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    #     x = torch.cat([front, x, end], dim=1)
    #     x = self.avg(x.permute(0, 2, 1))
    #     x = x.permute(0, 2, 1)
    #     return x
    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class Series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(Series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean


class Deep_Blocks(nn.Module):

    def __init__(self,input_size, forecast_horizon,num_blocks,num_fc_layers, expansion_dim,dropout,device):
        super(Deep_Blocks, self).__init__()
        self.device = device
        self.num_blocks = num_blocks
        self.forecast_horizon = forecast_horizon
        self.blocks = nn.ModuleList([Deep_Block(input_size=input_size,forecast_horizon=forecast_horizon,
                                                     num_fc_layers= num_fc_layers,expansion_dim= expansion_dim,dropout= dropout,device=device)
                                         for _ in range(self.num_blocks)])
    def forward(self, x,x_dec=None):
        forecasts = []
        #mean_coeff = 1 / self.num_blocks
        # forecast = tf.convert_to_tensor(0.0)
        rnd_numbers = np.random.randint(1, 5, self.num_blocks)
        for idx,block in enumerate(self.blocks):
            #block_forecasts = block(x)  # [B,past_h,channels]
            block_forecasts = block(x,-rnd_numbers[idx])  # [B,past_h,channels]
            #block_forecasts = block_forecasts * mean_coeff
            forecasts = block_forecasts

        return forecasts

#Original forecasting model
# class Deep_Block(nn.Module):
#
#     def __init__(self,input_size, forecast_horizon,num_fc_layers, expansion_dim,dropout,device):
#         super(Deep_Block, self).__init__()
#         self.device = device
#         self.input_size = input_size
#         self.forecast_horizon = forecast_horizon
#         self.num_fc_layers = num_fc_layers
#         self.fc_layers = nn.ModuleList([nn.Linear(in_features=input_size[0], out_features=expansion_dim)] +
#                       [nn.Linear(in_features=expansion_dim, out_features=self.forecast_horizon) for _ in range(self.num_fc_layers -1)])
#
#         self.dense_layer =  nn.Linear(input_size[0],self.forecast_horizon)
#         self.skip_layer = nn.Linear(input_size[0], self.forecast_horizon)
#         self.output_layer = nn.Linear(self.forecast_horizon, self.forecast_horizon)
#
#         self.relu = nn.ReLU()
#         self.dropout= nn.Dropout(dropout)
#
#         self.n_linear = N_Linear(input_size=(self.input_size[0],self.input_size[1]),
#                                   forecast_horizon=self.forecast_horizon,individual=False)
#
#     def forward(self, x,number):
#         x = self.n_linear(x,number)
#
#         return x


class Deep_Block(nn.Module):

    def __init__(self,input_size, forecast_horizon,num_fc_layers, expansion_dim,dropout,device):
        super(Deep_Block, self).__init__()
        self.device = device
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.num_fc_layers = num_fc_layers
        self.fc_layers = nn.ModuleList([nn.Linear(in_features=input_size[0], out_features=expansion_dim)] +
                      [nn.Linear(in_features=expansion_dim, out_features=expansion_dim) for _ in range(self.num_fc_layers -1)])
        self.dense_layer =  nn.Linear(expansion_dim,self.forecast_horizon)
        self.skip_layer = nn.Linear(input_size[0], self.forecast_horizon)
        self.output_layer = nn.Linear(self.forecast_horizon, self.forecast_horizon)

        self.relu = nn.ReLU()
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        #seq_last = x[:, -1:, :].detach()  # [B,1,num_series]
        seq_last = x[:, number, :].detach().unsqueeze(1)
        x = x - seq_last
        forecasts = torch.zeros(x.shape[0], self.forecast_horizon, x.shape[2]).to(self.device)
        for series in range(x.shape[2]):
            input = x[:, :, series]  ##[B,time_dim]
            ts = input
            for layer in self.fc_layers:  # to [B,channels, hidden_dim]
                ts = layer(ts)
                #ts = self.relu(ts)
                ts = self.dropout(ts)
            ts = self.dense_layer(ts)      # [B,forecast_dim]
            ts = ts + self.skip_layer(input)  # skip connction (B,forecast_dim)
            ts = self.output_layer(ts)  # (B,forecast_dim)
            forecasts[:, :, series] = ts
        forecasts = forecasts + seq_last


        return forecasts



#Forecasting model variant
class DC_BEATS(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            kernel_size: int,
            num_stacks: int,
            num_blocks: int,
            fc_hidden_layers:int,
            fc_hidden_units:int,
            block_sharing: bool,
            dropout: float,
            device:str
    ):
        super(DC_BEATS, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.fc_hidden_layers = fc_hidden_layers
        self.fc_hidden_units = fc_hidden_units
        self.block_sharing = block_sharing
        self.dropout = dropout
        self.device = device

        self.stacks = nn.ModuleList([
            DC_Stack(
                input_size=self.input_size,
                kernel_size=self.kernel_size,
                forecast_horizon=output_size,
                num_blocks=self.num_blocks,
                fc_hidden_layers = self.fc_hidden_layers,
                fc_hidden_units =  self.fc_hidden_units,
                block_sharing=self.block_sharing,
                dropout=self.dropout,
                device=device
            )
            for i in range(num_stacks)
        ])

        self.ll = nn.Linear(input_size[0],output_size)
    def forward(self, res,x_dec=None):
        residuals = res
        forecasts = 0.0
        seq_last = res[:, -1:, :].detach()
        x = res - seq_last

        mean_coeff = 1 / self.num_stacks
        # forecast = tf.convert_to_tensor(0.0)
        for i, _ in enumerate(range(len(self.stacks))):
            residuals, stack_forecast = self.stacks[i](residuals)  #[B,past_h,channels]
            stack_forecast = stack_forecast * mean_coeff
            forecasts = forecasts * mean_coeff
            forecasts = forecasts + stack_forecast

        forecasts = forecasts + seq_last

        return forecasts

class DC_Stack(nn.Module):
    def __init__(
            self,
            input_size: tuple,
            kernel_size: int,
            forecast_horizon: int,
            num_blocks: int,
            fc_hidden_layers:int,
            fc_hidden_units:int,
            block_sharing: bool,
            dropout: float,
            device:str
    ):
        super(DC_Stack, self).__init__()
        self.kernel_size = kernel_size
        self.fc_hidden_layers = fc_hidden_layers
        self.fc_hidden_units = fc_hidden_units
        self.num_blocks = num_blocks
        self.device = device

        self.blocks = nn.ModuleList([DC_Block(input_size=input_size, forecast_horizon=forecast_horizon, num_fc_layers=fc_hidden_layers ,expansion_dim=fc_hidden_units,dropout=dropout,device=device)])

        for i in range(1, num_blocks):
            if block_sharing:
                self.blocks.append(self.blocks[0])
            else:
                self.blocks.append(DC_Block(input_size=input_size, forecast_horizon=forecast_horizon,num_fc_layers=fc_hidden_layers ,expansion_dim=fc_hidden_units,dropout=dropout,device=device ))

    def forward(self, residuals):
        mean_coeff = 1 / self.num_blocks
        forecast = torch.tensor(0.0).to(self.device)
        for _, block in enumerate(self.blocks):
            backcast, forecast_block = block(residuals)
            residuals = residuals - backcast  # calculate resisual
            forecast_block = forecast_block * mean_coeff
            forecast = forecast * mean_coeff
            forecast = forecast + forecast_block

        return residuals, forecast  # * level


class DC_Block(nn.Module):

    def __init__(self, input_size, forecast_horizon, num_fc_layers, expansion_dim, dropout,device):
        super(DC_Block, self).__init__()
        self.input_size = input_size
        self.forecast_horizon = forecast_horizon
        self.num_fc_layers = num_fc_layers
        self.expansion_dim = expansion_dim #hidden units
        self.device = device
        self.fc_layers = nn.ModuleList([nn.Linear(in_features=self.input_size[0], out_features=self.expansion_dim)] +
                                       [nn.Linear(in_features=self.expansion_dim, out_features=self.expansion_dim) for _ in
                                        range(self.num_fc_layers - 1)])
        self.dense_layer = nn.Linear(self.expansion_dim, self.forecast_horizon)
        self.skip_layer = nn.Linear(self.input_size[0], self.forecast_horizon)
        self.output_layer = nn.Linear(self.forecast_horizon, self.forecast_horizon)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.9)
        #self.decomposition = Series_decomp(1)
        self.decomposition = series_decomp_multi([1,2,24,25])


    def forward(self, x):

        residuals, x_trend = self.decomposition(x)  ## to [B,time_dim,channels]

        forecasts = torch.zeros(x.shape[0], self.forecast_horizon, x.shape[2]).to(self.device)
        for series in range(x.shape[2]):
            input = x_trend[:, :, series]  ##[B,time_dim]
            ts = input
            for layer in self.fc_layers:  # to [B,channels, hidden_dim]
                ts = layer(ts)
            ts = self.dropout(self.dense_layer(ts))  # [B,forecast_dim]
            # ts = self.relu(ts)            #relu
            # ts = self.dropout(ts)         # dropout
            ts = self.dropout(ts + self.skip_layer(input))  # skip connction (B,forecast_dim)
            ts = self.dropout(self.output_layer(ts))  # (B,forecast_dim)
            forecasts[:, :, series] = ts

        return residuals, forecasts




#GRU in pytorch
# #input [B,L(sequence_len or time-steps),input_size or num_series]
# input = torch.randn(5, 3, 10)
# rnn = nn.GRU(input_size=10, hidden_size=20, num_layers=2,batch_first=True)
# #h0 [num_layers,B,H_out]
# h0 = torch.randn(2, 5, 20)
# #output = [B,L,H_out]
# output, h0 = rnn(input, h0) #[5,3,20] & [2,5,20]

# 3 sequence_length or time-steps) equals 3 GRU CELL, with each cell producing 20 values(hidden_size)
#Each batch goes through all cells, thus producing (3 * 20) for each batch, which are return in "out" variable
#Final output of all batches 5, equals (5,3,20) => output variable
#in h0 variable, only the output of the last cell (20 values) is returned for each batch, example if 1 layer -> h0 = [1,5,20]
#in other words, h0 is a subset/chunck of output, that is, h0 is equal to output[:,-1,:]
