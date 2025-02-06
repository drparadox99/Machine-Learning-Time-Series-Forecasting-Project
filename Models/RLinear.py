import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.individual = False
        self.forecast_horizon = configs.forecast_horizon
        self.num_series = configs.num_series
        self.Linear = nn.ModuleList([
            nn.Linear(configs.past_history, configs.forecast_horizon) for _ in range(configs.num_series)
        ]) if self.individual else nn.Linear(configs.past_history, configs.forecast_horizon)

        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.num_series)
        self.device = configs.device

    def forward(self, x,x_dec):
        # x: [B, lookback_w, num_series]
        x = self.rev(x, 'norm')
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros([x.size(0), self.forecast_horizon, self.num_series], dtype=x.dtype).to(device)
            #pred = torch.zeros_like((x.shape[0],self.forecast_horizon,self.num_series))
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm')
        return pred