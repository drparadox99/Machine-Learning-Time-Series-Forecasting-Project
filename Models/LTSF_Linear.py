
import torch
import torch.nn as nn


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

class DLinearModel(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, kernel_size, seq_len: int, pred_len: int, individual: bool, num_series,device):
        super(DLinearModel, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Decompsition Kernel Size
        self.decompsition = Series_decomp(kernel_size)
        self.individual = individual
        self.num_series = num_series
        self.num_hidden_layers = 3
        self.fc_hidden_layers_s = nn.ModuleList([
            nn.Linear(self.seq_len, self.seq_len)
            for i in range(self.num_hidden_layers)
        ])
        self.fc_hidden_layers_t = nn.ModuleList([
            nn.Linear(self.seq_len, self.seq_len)
            for i in range(self.num_hidden_layers)
        ])
        self.dropout = nn.Dropout(0.2)
        self.dense =  nn.Linear(self.pred_len, self.pred_len)

        # individual: a linear layer for each variate(channel) individually'
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.num_series):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x,x_dec=None):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)

        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0,2,1)  # permute to [Batch,num_series,seq_len]
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(
                seasonal_init.self.device)  # [Batch,num_series,seq_len]

            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.self.device)  # [Batch,num_series,seq_len]

            for i in range(self.num_series):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])  # [Batch,num_series,forecast horizon]
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])  # [Batch,num_series,forecast horizon]
        else:

            seasonal_output = self.Linear_Seasonal(seasonal_init) #[Batch,num_series,forecast horizon]
            trend_output = self.Linear_Trend(trend_init)
            if seasonal_output.ndim == 1 and trend_output.ndim == 1:  ##(return 2D tensor)  add channel dim if dim 1
                seasonal_output = seasonal_output.unsqueeze(0)
                trend_output = trend_output.unsqueeze(0)

        x = seasonal_output + trend_output  # [Batch,num_series,forecast_horizon]

        if x.ndim == 2:  # (return 3D tensor)  if x dim_2 (one batch) add batch dim
            x = x.unsqueeze(0)
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        # return x # to [Batch,Channel, Output length, ]

class Simple_Linear(nn.Module):

    def __init__(self,input_size, forecasting_horizon,device):
        super(Simple_Linear, self).__init__()
        self.device = device
        self.input_size = input_size
        self.forecasting_horizon = forecasting_horizon
        self.linear = nn.Linear(self.input_size[0], self.forecasting_horizon)

    def forward(self, x,x_dec=None):
        foreacasts = torch.zeros(x.shape[0],self.forecasting_horizon,x.shape[2]).to(self.device)
        for series in range(x.shape[2]):
            foreacasts[:,:,series] = self.linear(x[:,:,series])
            #foreacasts[:, :, series] = self.fc_layers_series[series](x[:,:,series])
        return foreacasts


class N_Linear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, args, individual=False):
        super(N_Linear, self).__init__()
        self.device = args.device
        self.seq_len = args.past_history
        self.pred_len = args.forecast_horizon

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = args.num_series
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, x_dec=None):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach() # [B,1,num_series]  #
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(self.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + seq_last
        return x  # [Batch, Output length, Channel]
