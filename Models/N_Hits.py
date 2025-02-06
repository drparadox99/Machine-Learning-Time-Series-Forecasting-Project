




from typing import List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
import numpy as np

import torch as t
import torch.nn as nn
from typing import Tuple
from functools import partial




from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        num_layers: int,
        layer_width: int,
        nr_params: int,
        pooling_kernel_size: int,
        n_freq_downsample: int,
        batch_norm: bool,
        dropout: float,
        MaxPool1d: bool,
    ):

        super().__init__()

        self.num_layers = num_layers
        self.layer_width = layer_width
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.nr_params = nr_params
        self.pooling_kernel_size = pooling_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.MaxPool1d = MaxPool1d

        n_theta_backcast = max(input_chunk_length // n_freq_downsample, 1)
        n_theta_forecast = max(output_chunk_length // n_freq_downsample, 1)

        # entry pooling layer
        pool1d = nn.MaxPool1d if self.MaxPool1d else nn.AvgPool1d
        self.pooling_layer = pool1d(
            kernel_size=self.pooling_kernel_size,
            stride=self.pooling_kernel_size,
            ceil_mode=True,
        )

        # layer widths
        in_len = int(np.ceil(input_chunk_length / pooling_kernel_size))
        self.layer_widths = [in_len] + [self.layer_width] * self.num_layers

        # FC layers
        layers = []
        for i in range(self.num_layers):
            layers.append(
                nn.Linear(
                    in_features=self.layer_widths[i],
                    out_features=self.layer_widths[i + 1],
                )
            )
            #layers.append(self.activation)

            if self.batch_norm:
                layers.append(nn.BatchNorm1d(num_features=self.layer_widths[i + 1]))

            # if self.dropout > 0: nn.Dropout(dropout)
            #     layers.append(MonteCarloDropout(p=self.dropout))

        self.layers = nn.Sequential(*layers)

        # Fully connected layer producing forecast/backcast expansion coefficients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood for the forecast.
        self.backcast_linear_layer = nn.Linear(
            in_features=layer_width, out_features=n_theta_backcast
        )
        self.forecast_linear_layer = nn.Linear(
            in_features=layer_width, out_features=nr_params * n_theta_forecast
        )

    def forward(self, x):
        print("x ", x.shape)
        batch_size = x.shape[0]

        # pooling
        #x = x.unsqueeze(1)
        x = self.pooling_layer(x)
        x = x.squeeze(1)
        # fully connected layer stack
        x = self.layers(x)

        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # set the expansion coefs in last dimension for the forecasts
        theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)

        # interpolate function expects (batch, "channels", time)
        theta_backcast = theta_backcast.unsqueeze(1)

        # interpolate both backcast and forecast from the thetas
        x_hat = F.interpolate(
            theta_backcast, size=self.input_chunk_length, mode="linear"
        )
        y_hat = F.interpolate(
            theta_forecast, size=self.output_chunk_length, mode="linear"
        )

        x_hat = x_hat.squeeze(1)  # remove 2nd dim we added before interpolation

        # Set the distribution parameters as the last dimension
        y_hat = y_hat.reshape(x.shape[0], self.output_chunk_length, self.nr_params)

        return x_hat, y_hat

#
# class _Stack(nn.Module):
#     def __init__(
#         self,
#         input_chunk_length: int,
#         output_chunk_length: int,
#         num_blocks: int,
#         num_layers: int,
#         layer_width: int,
#         nr_params: int,
#         pooling_kernel_sizes: int,
#         n_freq_downsample: int,
#         batch_norm: bool,
#         dropout: float,
#         activation: str,
#         MaxPool1d: bool,
#     ):
#
#         super().__init__()
#
#         self.input_chunk_length = input_chunk_length
#         self.output_chunk_length = output_chunk_length
#         self.nr_params = nr_params
#
#         self.blocks_list = [
#             _Block(
#                 input_chunk_length,
#                 output_chunk_length,
#                 num_layers,
#                 layer_width,
#                 nr_params,
#                 pooling_kernel_sizes,
#                 n_freq_downsample,
#                 batch_norm=(
#                     batch_norm and i == 0
#                 ),  # batch norm only on first block of first stack
#                 dropout=dropout,
#                 activation=activation,
#                 MaxPool1d=MaxPool1d,
#             )
#             for i in range(num_blocks)
#         ]
#         self.blocks = nn.ModuleList(self.blocks_list)
#
#     def forward(self, x):
#         # One forecast vector per parameter in the distribution
#         stack_forecast = torch.zeros(
#             x.shape[0],
#             self.output_chunk_length,
#             self.nr_params,
#             device=x.device,
#             dtype=x.dtype,
#         )
#
#         for block in self.blocks_list:
#             # pass input through block
#             x_hat, y_hat = block(x)
#
#             # add block forecast to stack forecast
#             stack_forecast = stack_forecast + y_hat
#
#             # subtract backcast from input to produce residual
#             x = x - x_hat
#
#         stack_residual = x
#
#         return stack_residual, stack_forecast
#
#
# class _NHiTSModule(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         nr_params: int,
#         num_stacks: int,
#         num_blocks: int,
#         num_layers: int,
#         layer_widths: List[int],
#         pooling_kernel_sizes: Tuple[Tuple[int]],
#         n_freq_downsample: Tuple[Tuple[int]],
#         batch_norm: bool,
#         dropout: float,
#         activation: str,
#         MaxPool1d: bool,
#     ):
#
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.nr_params = nr_params
#
#
#         self.stacks_list = [
#             _Stack(
#                 self.input_dim,
#                 self.output_dim,
#                 num_blocks,
#                 num_layers,
#                 layer_widths[i],
#                 nr_params,
#                 pooling_kernel_sizes[i],
#                 n_freq_downsample[i],
#                 batch_norm=(
#                     batch_norm and i == 0
#                 ),  # batch norm only on first block of first stack
#                 dropout=dropout,
#                 activation=activation,
#                 MaxPool1d=MaxPool1d,
#             )
#             for i in range(num_stacks)
#         ]
#
#         self.stacks = nn.ModuleList(self.stacks_list)
#
#         # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
#         # backpropagated). Removing this line would cause logtensorboard to crash, since no gradient is stored
#         # on this params (the last block backcast is not part of the final output of the net).
#         self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)
#
#     def forward(self, x: Tuple):
#
#         for stack in self.stacks_list:
#             # compute stack output
#             stack_residual, stack_forecast = stack(x)
#
#             # add stack forecast to final output
#             y = y + stack_forecast
#
#             # set current stack residual as input for next stack
#             x = stack_residual
#
#         # In multivariate case, we get a result [x1_param1, x1_param2], [y1_param1, y1_param2], [x2..], [y2..], ...
#         # We want to reshape to original format. We also get rid of the covariates and keep only the target dimensions.
#         # The covariates are by construction added as extra time series on the right side. So we need to get rid of this
#         # right output (keeping only :self.output_dim).
#         y = y.view(
#             y.shape[0], self.output_chunk_length, self.input_dim, self.nr_params
#         )[:, :, : self.output_dim, :]
#
#         return y





#
#
# #https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/nhits/sub_modules.py
# # Cell
# class _StaticFeaturesEncoder(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(_StaticFeaturesEncoder, self).__init__()
#         layers = [nn.Dropout(p=0.5),
#                   nn.Linear(in_features=in_features, out_features=out_features),
#                   nn.ReLU()]
#         self.encoder = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x
#
# class RepeatVector(nn.Module):
#     """
#     Receives x input of dim [N,C], and repeats the vector
#     to create tensor of shape [N, C, K]
#     : repeats: int, the number of repetitions for the vector.
#     """
#     def __init__(self, repeats):
#         super(RepeatVector, self).__init__()
#         self.repeats = repeats
#
#     def forward(self, x):
#         x = x.unsqueeze(-1).repeat(1, 1, self.repeats) # <------------ Mejorar?
#         return x
# class _sEncoder(nn.Module):
#     def __init__(self, in_features, out_features, n_time_in):
#         super(_sEncoder, self).__init__()
#         layers = [nn.Dropout(p=0.5),
#                   nn.Linear(in_features=in_features, out_features=out_features),
#                   nn.ReLU()]
#         self.encoder = nn.Sequential(*layers)
#         self.repeat = RepeatVector(repeats=n_time_in)
#
#     def forward(self, x):
#         # Encode and repeat values to match time
#         x = self.encoder(x)
#         x = self.repeat(x) # [N,S_out] -> [N,S_out,T]
#         return x
#
# ACTIVATIONS = ['ReLU',
#                'Softplus',
#                'Tanh',
#                'SELU',
#                'LeakyReLU',
#                'PReLU',
#                'Sigmoid']
#
# class _NHITSBlock(nn.Module):
#     """
#     N-HiTS block which takes a basis function as an argument.
#     """
#     def __init__(self, n_time_in: int, n_time_out: int, n_x: int,
#                  n_s: int, n_s_hidden: int, n_theta: int, n_theta_hidden: list,
#                  n_pool_kernel_size: int, pooling_mode: str, basis: nn.Module,
#                  n_layers: int,  batch_normalization: bool, dropout_prob: float, activation: str):
#         """
#         """
#         super().__init__()
#
#         assert (pooling_mode in ['max','average'])
#
#         n_time_in_pooled = int(np.ceil(n_time_in/n_pool_kernel_size))
#
#         if n_s == 0:
#             n_s_hidden = 0
#         n_theta_hidden = [n_time_in_pooled + (n_time_in+n_time_out)*n_x + n_s_hidden] + n_theta_hidden
#
#         self.n_time_in = n_time_in
#         self.n_time_out = n_time_out
#         self.n_s = n_s
#         self.n_s_hidden = n_s_hidden
#         self.n_x = n_x
#         self.n_pool_kernel_size = n_pool_kernel_size
#         self.batch_normalization = batch_normalization
#         self.dropout_prob = dropout_prob
#
#         assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
#         activ = getattr(nn, activation)()
#
#         if pooling_mode == 'max':
#             self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
#                                               stride=self.n_pool_kernel_size, ceil_mode=True)
#         elif pooling_mode == 'average':
#             self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
#                                               stride=self.n_pool_kernel_size, ceil_mode=True)
#
#         hidden_layers = []
#         for i in range(n_layers):
#             hidden_layers.append(nn.Linear(in_features=n_theta_hidden[i], out_features=n_theta_hidden[i+1]))
#             hidden_layers.append(activ)
#
#             if self.batch_normalization:
#                 hidden_layers.append(nn.BatchNorm1d(num_features=n_theta_hidden[i+1]))
#
#             if self.dropout_prob>0:
#                 hidden_layers.append(nn.Dropout(p=self.dropout_prob))
#
#         output_layer = [nn.Linear(in_features=n_theta_hidden[-1], out_features=n_theta)]
#         layers = hidden_layers + output_layer
#
#         # n_s is computed with data, n_s_hidden is provided by user, if 0 no statics are used
#         if (self.n_s > 0) and (self.n_s_hidden > 0):
#             self.static_encoder = _StaticFeaturesEncoder(in_features=n_s, out_features=n_s_hidden)
#         self.layers = nn.Sequential(*layers)
#         self.basis = basis
#
#     def forward(self, insample_y: t.Tensor, insample_x_t: t.Tensor,
#                 outsample_x_t: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
#
#         insample_y = insample_y.unsqueeze(1)
#         # Pooling layer to downsample input
#         insample_y = self.pooling_layer(insample_y)
#         insample_y = insample_y.squeeze(1)
#
#         batch_size = len(insample_y)
#         if self.n_x > 0:
#             insample_y = t.cat(( insample_y, insample_x_t.reshape(batch_size, -1) ), 1)
#             insample_y = t.cat(( insample_y, outsample_x_t.reshape(batch_size, -1) ), 1)
#
#         # Static exogenous
#         if (self.n_s > 0) and (self.n_s_hidden > 0):
#             x_s = self.static_encoder(x_s)
#             insample_y = t.cat((insample_y, x_s), 1)
#
#         # Compute local projection weights and projection
#         theta = self.layers(insample_y)
#         backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)
#
#         return backcast, forecast
#
# # Cell
# class _NHITS(nn.Module):
#     """
#     N-HiTS Model.
#     """
#     def __init__(self,
#                  n_time_in,
#                  n_time_out,
#                  n_s,
#                  n_x,
#                  n_s_hidden,
#                  n_x_hidden,
#                  stack_types: list,
#                  n_blocks: list,
#                  n_layers: list,
#                  n_theta_hidden: list,
#                  n_pool_kernel_size: list,
#                  n_freq_downsample: list,
#                  pooling_mode,
#                  interpolation_mode,
#                  dropout_prob_theta,
#                  activation,
#                  initialization,
#                  batch_normalization,
#                  shared_weights):
#         super().__init__()
#
#         self.n_time_out = n_time_out
#
#         blocks = self.create_stack(stack_types=stack_types,
#                                    n_blocks=n_blocks,
#                                    n_time_in=n_time_in,
#                                    n_time_out=n_time_out,
#                                    n_x=n_x,
#                                    n_x_hidden=n_x_hidden,
#                                    n_s=n_s,
#                                    n_s_hidden=n_s_hidden,
#                                    n_layers=n_layers,
#                                    n_theta_hidden=n_theta_hidden,
#                                    n_pool_kernel_size=n_pool_kernel_size,
#                                    n_freq_downsample=n_freq_downsample,
#                                    pooling_mode=pooling_mode,
#                                    interpolation_mode=interpolation_mode,
#                                    batch_normalization=batch_normalization,
#                                    dropout_prob_theta=dropout_prob_theta,
#                                    activation=activation,
#                                    shared_weights=shared_weights,
#                                    initialization=initialization)
#         self.blocks = t.nn.ModuleList(blocks)
#
#     def create_stack(self, stack_types, n_blocks,
#                      n_time_in, n_time_out,
#                      n_x, n_x_hidden, n_s, n_s_hidden,
#                      n_layers, n_theta_hidden,
#                      n_pool_kernel_size, n_freq_downsample, pooling_mode, interpolation_mode,
#                      batch_normalization, dropout_prob_theta,
#                      activation, shared_weights, initialization):
#
#         block_list = []
#         for i in range(len(stack_types)):
#             #print(f'| --  Stack {stack_types[i]} (#{i})')
#             for block_id in range(n_blocks[i]):
#
#                 # Batch norm only on first block
#                 if (len(block_list)==0) and (batch_normalization):
#                     batch_normalization_block = True
#                 else:
#                     batch_normalization_block = False
#
#                 # Shared weights
#                 if shared_weights and block_id>0:
#                     nbeats_block = block_list[-1]
#                 else:
#                     if stack_types[i] == 'identity':
#                         n_theta = (n_time_in + max(n_time_out//n_freq_downsample[i], 1) )
#                         basis = IdentityBasis(backcast_size=n_time_in,
#                                               forecast_size=n_time_out,
#                                               interpolation_mode=interpolation_mode)
#
#                     else:
#                         assert 1<0, f'Block type not found!'
#
#                     nbeats_block = _NHITSBlock(n_time_in=n_time_in,
#                                                    n_time_out=n_time_out,
#                                                    n_x=n_x,
#                                                    n_s=n_s,
#                                                    n_s_hidden=n_s_hidden,
#                                                    n_theta=n_theta,
#                                                    n_theta_hidden=n_theta_hidden[i],
#                                                    n_pool_kernel_size=n_pool_kernel_size[i],
#                                                    pooling_mode=pooling_mode,
#                                                    basis=basis,
#                                                    n_layers=n_layers[i],
#                                                    batch_normalization=batch_normalization_block,
#                                                    dropout_prob=dropout_prob_theta,
#                                                    activation=activation)
#
#                 # Select type of evaluation and apply it to all layers of block
#                 init_function = partial(init_weights, initialization=initialization)
#                 nbeats_block.layers.apply(init_function)
#                 #print(f'     | -- {nbeats_block}')
#                 block_list.append(nbeats_block)
#         return block_list
#
#     def forward(self, S: t.Tensor, Y: t.Tensor, X: t.Tensor,
#                 insample_mask: t.Tensor, outsample_mask: t.Tensor,
#                 return_decomposition: bool=False):
#
#         # insample
#         insample_y    = Y[:, :-self.n_time_out]
#         insample_x_t  = X[:, :, :-self.n_time_out]
#         insample_mask = insample_mask[:, :-self.n_time_out]
#
#         # outsample
#         outsample_y   = Y[:, -self.n_time_out:]
#         outsample_x_t = X[:, :, -self.n_time_out:]
#         outsample_mask = outsample_mask[:, -self.n_time_out:]
#
#         if return_decomposition:
#             forecast, block_forecasts = self.forecast_decomposition(insample_y=insample_y,
#                                                                     insample_x_t=insample_x_t,
#                                                                     insample_mask=insample_mask,
#                                                                     outsample_x_t=outsample_x_t,
#                                                                     x_s=S)
#             return outsample_y, forecast, block_forecasts, outsample_mask
#
#         else:
#             forecast = self.forecast(insample_y=insample_y,
#                                      insample_x_t=insample_x_t,
#                                      insample_mask=insample_mask,
#                                      outsample_x_t=outsample_x_t,
#                                      x_s=S)
#             return outsample_y, forecast, outsample_mask
#
#     def forecast(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
#                  outsample_x_t: t.Tensor, x_s: t.Tensor):
#
#         residuals = insample_y.flip(dims=(-1,))
#         insample_x_t = insample_x_t.flip(dims=(-1,))
#         insample_mask = insample_mask.flip(dims=(-1,))
#
#         forecast = insample_y[:, -1:] # Level with Naive1
#         for i, block in enumerate(self.blocks):
#             backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
#                                              outsample_x_t=outsample_x_t, x_s=x_s)
#             residuals = (residuals - backcast) * insample_mask
#             forecast = forecast + block_forecast
#
#         return forecast
#
#     def forecast_decomposition(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
#                                outsample_x_t: t.Tensor, x_s: t.Tensor):
#
#         residuals = insample_y.flip(dims=(-1,))
#         insample_x_t = insample_x_t.flip(dims=(-1,))
#         insample_mask = insample_mask.flip(dims=(-1,))
#
#         n_batch, n_channels, n_t = outsample_x_t.size(0), outsample_x_t.size(1), outsample_x_t.size(2)
#
#         level = insample_y[:, -1:] # Level with Naive1
#         block_forecasts = [ level.repeat(1, n_t) ]
#
#         forecast = level
#         for i, block in enumerate(self.blocks):
#             backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
#                                              outsample_x_t=outsample_x_t, x_s=x_s)
#             residuals = (residuals - backcast) * insample_mask
#             forecast = forecast + block_forecast
#             block_forecasts.append(block_forecast)
#
#         # (n_batch, n_blocks, n_t)
#         block_forecasts = t.stack(block_forecasts)
#         block_forecasts = block_forecasts.permute(1,0,2)
#
#         return forecast, block_forecasts
#
