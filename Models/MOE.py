import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn
from Models.LTSF_Linear import N_Linear
from Models.Formers.VanillaTransformer.VanillaTransformer import Model as VanillaTransformer
from Models.BEATS_CELL import BEATS_CELL




class N_Linear_MOE(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, args, individual=False):
        super(N_Linear_MOE, self).__init__()
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
        #seq_last = x[:, -1:, :].detach() # [B,1,num_series]  #
        seq_last = torch.max(x, dim=1).values.unsqueeze(1) #get columns' highest values
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
