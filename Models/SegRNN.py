import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # remove this, the performance will be bad
        self.lucky = nn.Embedding(7, 256// 2)

        self.seq_len = configs.past_history
        self.pred_len = configs.forecast_horizon
        self.enc_in = configs.num_series  #num_series
        self.patch_len = 16
        self.d_model = 256

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.dropout = nn.Dropout(configs.dropout)
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x,x_dec=None):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        B, L, C = x.shape  #[4860,288,8]
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        #enc_in = self.relu(xd)
        enc_in = xd
        enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d
        dec_out = self.gru(dec_in, enc_out)[0]  # B * C * M, 1, d
        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H

        y = y + seq_last

        return y

#
# import torch
# import torch.nn as nn
# import math
#
#
# class Model(nn.Module):
#     def __init__(self, num_series, looback_w, fh):
#         super(Model, self).__init__()
#
#         # remove this, the performance will be bad
#         self.lucky = nn.Embedding(7, 256 // 2)
#
#         self.seq_len = lookback_w  # looback_w
#         self.pred_len = fh  # pred_len
#         self.enc_in = num_series  # num_series
#         self.patch_len = 16  # patch_len
#         self.d_model = 256  # d_model
#
#         self.linear_patch = nn.Linear(self.patch_len, self.d_model)  # shape 16 -> 256
#         self.relu = nn.ReLU()
#
#         self.gru = nn.GRU(  # from 256 to 256
#             input_size=self.d_model,
#             hidden_size=self.d_model,
#             num_layers=1,
#             bias=True,
#             batch_first=True,
#         )
#
#         self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))  # shape [6,128]
#         self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))  # shape [8,128]
#         self.dropout = nn.Dropout(0.0)
#         self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)  # shape 256 -> 16
#
#     def forward(self, x):
#         seq_last = x[:, -1:, :].detach()  # shape [4860,1,8]
#         x = x - seq_last
#         B, L, C = x.shape  # shape [4860,288,8]
#         N = self.seq_len // self.patch_len  # N=18
#         M = self.pred_len // self.patch_len  # M=6
#         W = self.patch_len  # W=16
#         d = self.d_model  # d=256
#
#         xw = x.permute(0, 2, 1).reshape(B * C, N,
#                                         -1)  # B, L, C -> B, C, L -> B * C, N, W => # (4860,288,8-> 4860,8,288->4860*8,18,16)  #each batch represents a chopped up time-series of (N,w)
#         xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d  #from [4860*8, 18, 16] to [4860*8(38880), 18, 256]
#         # enc_in = self.relu(xd)
#         enc_in = xd
#
#         enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1,
#                                                            self.d_model)  # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d
#         # output, h0 = shape:[4860*8(38880), 18, 256],[1, 4860*8(38880), 256]
#         # repeat = from [1,4860*8,256]  to from [1,4860*8,1536(256*M)]
#         # view = from [1,4860*8,1536(256*M)] to [1,4860*8*M,256]
#
#         dec_in = torch.cat([
#             self.pos_emb.unsqueeze(0).repeat(B * C, 1, 1),  # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
#             self.channel_emb.unsqueeze(1).repeat(B, M, 1)  # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
#         ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d -> B * C * M, 1, d
#
#         # pos_emb: shape[6, 128]
#         # channel_em: shape[8,128]
#         # unsqueeze pos_emb : from [6,128] to [1, 6, 128]
#         # unsqueeze channel_emb : from [8,128] to [8,1,128]
#         # repeat pos_emb (B*C, 1, 1) : from [6,128] to [(4860 * 8)38880, 6, 128]
#         # repeat channel_emb (B, M, 1) : from [8,1,128] to [(4860 * 8)38880, 6, 128]
#         # cat: from [(4860 * 8)38880, 6, 128] & [(4860 * 8)38880, 6, 128] to [(4860 * 8)38880, 6, 256]
#         # flatten:  torch.Size([(4860 * 8 * M)233280, 256])
#         # dec_in: [233280, 1, 256] (hidden state shape for decoder (B,time_dim,dim) and h0 (1(num_layers),B,dim)
#
#         dec_out = self.gru(dec_in, enc_out)[0]  # B * C * M, 1, d
#         # dec_out: [(4860 * 8 * M)233280,1,256]
#         yd = self.dropout(dec_out)
#         yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W  # [(4860 * 8 * M)233280,1,16]
#         y = yw.reshape(B, C, -1).permute(0, 2, 1)  # B, C, H
#         # y shape: [4860,8,96]  # y shape:[4860,96,8]
#         y = y + seq_last
#         return y
#
#
# fh = 96
# lookback_w = 288
# num_series = 8
#
# x = torch.rand(4860, 288, 8)
# model = Model(num_series, lookback_w, fh)
# model(x)
