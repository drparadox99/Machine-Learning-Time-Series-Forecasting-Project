import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Formers.MinusFormer.layers.Transformer_EncDec import Encoder, EncoderLayer
from Models.Formers.MinusFormer.layers.SelfAttention_Family import FullAttention, AttentionLayer, FlashAttention, ProbAttention
#from layers.Embed import DataEmbedding_inverted
import numpy as np
from Models.Formers.MinusFormer.utils.tools import standard_scaler


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2402.02332
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.forecast_horizon
        self.embed = nn.Linear(configs.num_series, configs.d_model)
        self.backbone = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model,
                        configs.n_heads) if True else None,
                    configs.d_model,
                    configs.forecast_horizon,
                    128,
                    dropout=configs.dropout,
                    gate=1
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        print('Minusformer ...')

    def forward(self, x_enc,x_dec=None):
        x  = x_enc
        x = x.permute(0, 2, 1)
        scaler = standard_scaler(x)
        x = scaler.transform(x)

        x_emb = self.embed(x.permute(0,2,1))
        output = self.backbone(x_emb)
        output = scaler.inverted(output[:, :x.size(1), :])
        return output.permute(0, 2, 1)
