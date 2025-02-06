import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Formers.Layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from Models.Formers.Layers.SelfAttention_Family import FullAttention, AttentionLayer,ProbAttention
from Models.Formers.Embed import DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np

from Utils.RevIN import RevIN

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.forecast_horizon
        self.output_attention = configs.output_attention

        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_temp(configs.num_series, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.num_series, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)

        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.num_series, configs.d_model, configs.embed,
                                                           configs.freq,
                                                           configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.num_series, configs.d_model, configs.embed,
                                                           configs.freq,
                                                           configs.dropout)
        #Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     [
        #         ConvLayer(
        #             configs.d_model
        #         ) for l in range(configs.e_layers - 1)
        #     ] if configs.distil else None,
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.num_series, bias=True)
        )
        self.revin_layer = RevIN(configs.num_series)


    def forward(self, x_enc, x_dec,enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc : [B,lookback_w,d_model],
        # x_dec : [B,lookback_w+48,num_series]
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # enc_out : [B,lookback_w,d_model]
        dec_out = self.dec_embedding(x_dec)
        #dec_out:[B,lookback_w+48,d_model],enc_out : [B,lookback_w,d_model]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            #output: [B,lookback_w+48,num_series]
            output = dec_out[:, -self.pred_len:, :]
            return output   # [B, L, D]