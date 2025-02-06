import einops
import torch
import torch.nn as nn

from Models.Mamba.pscan import pscan
import math
import torch.nn.functional as F

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()

        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        #v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v

class GatingMlpBlock(nn.Module):
    def __init__(self, d_model, seq_len, survival_prob):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_model, seq_len)
        self.proj_2 = nn.Linear(d_model//2, d_model)
        self.prob = survival_prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        # if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
        #     return x
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x= self.dropout(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x= self.dropout(x)
        return x + shorcut


class GMLP_2(nn.Module):
    def __init__(
        self,d_model,seq_len,forecast_horizon,n_blocks,prob_0_L=[1, 0.5]
    ):
        super().__init__()
        self.survival_probs = torch.linspace(prob_0_L[0], prob_0_L[1], n_blocks)
        self.blocks = nn.ModuleList(
            [GatingMlpBlock(d_model, seq_len, prob) for prob in self.survival_probs]
        )
        self.d_model = d_model
        print("N° of blocks: ", len(self.blocks))
        self.output_proj = nn.Linear(d_model, forecast_horizon)
        self.dropout = nn.Dropout(0.3)

        self.norm = nn.InstanceNorm1d(seq_len, affine=True)




        self.d_state = 16
        self.d_inner = 2 * self.d_model
        self.dt_rank = math.ceil(self.d_model / 16)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=True)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)


    def forward(self, x,x_dec=None):
        seq_last = x[:, -1:, :].detach() # [B,1,num_series]  #
        x = x - seq_last
        x = x.permute(0,2,1) #[B,num_series,lookback_w]
        for gmlp_block in self.blocks:
            #x = gmlp_block(x)
            x = F.gelu(self.ssm(x))
            x = self.dropout(x)
        x = self.output_proj(x)
        x = self.dropout(x) #(nouveauté)
        x = x.permute(0, 2, 1)
        x = x + seq_last
        return x

    def ssm(self, x):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state],dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)

        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        y = self.selective_scan(x, delta, A, B, C, D)
        output = y * z
        output = self.out_proj(output)
        return output

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
