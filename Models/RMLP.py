
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Utils.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.past_history, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.past_history)
        )
        self.projection = nn.Linear(configs.past_history, configs.forecast_horizon)
        # self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.num_series)

    def forward(self, x, y):
        # x: [B, L, D]
        x = self.rev(x, 'norm')
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        pred = self.projection(x.transpose(1, 2)).transpose(1, 2)
        output = self.rev(pred, 'denorm')
        return output

