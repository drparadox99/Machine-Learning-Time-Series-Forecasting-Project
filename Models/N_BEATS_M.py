import torch
import torch.nn as nn
import torch.nn.functional as F

class NBeatsBlock(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        output_size: int,
        theta_size: int,
        fc_block_layers: int,
        fc_hidden_units: int,
        dropout: float
    ):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.theta_size = theta_size
        
        self.fc_hidden_units = nn.ModuleList([
            nn.Linear(input_size[0], fc_hidden_units) for _ in range(fc_block_layers)
        ])

        self.theta_layer = nn.Linear(fc_hidden_units, self.input_size[0])

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs_ = inputs.permute(0, 2, 1)  # shape to num_series, samples
        x = inputs_
        for layer in self.fc_hidden_units:
            x = F.relu(layer(x))
            x = self.dropout(x)
        theta = self.theta_layer(x)  # shape last dim to num_series
        theta = theta.permute(0, 2, 1)  # reshape back to samples, num_series
        backcast, forecast = theta[:, :, :self.input_size[0]], theta[:, :, -self.output_size:]
        return backcast, forecast


class NBeatsStack(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        output_size: int,
        num_blocks: int,
        fc_block_layers: int,
        fc_hidden_units: int,
        block_sharing: bool,
        dropout: float,
    ):
        super(NBeatsStack, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size=input_size, output_size=output_size, theta_size=input_size[0] + output_size,
                        fc_block_layers=fc_block_layers, fc_hidden_units=fc_hidden_units, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.output_size = output_size
        self.block_sharing = block_sharing

    def forward(self, residuals):
        forecast = 0
        for block in self.blocks:
            backcast, forecast_block = block(residuals)
            forecast = forecast_block + forecast
            residuals =  residuals - backcast
        return residuals, forecast




class NBeats_M(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        output_size: int,
        num_stacks: int,
        num_blocks: int,
        fc_block_layers: int,
        fc_hidden_units: int,
        block_sharing: bool,
        dropout: float
    ):
        super(NBeats_M, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.fc_block_layers = fc_block_layers
        self.fc_hidden_units = fc_hidden_units
        self.block_sharing = block_sharing
        self.dropout = dropout
        
        # create first stack with one block
        self.nbeats_stack = NBeatsStack(
            input_size=self.input_size,
            output_size=self.output_size,
            num_blocks=self.num_blocks,
            fc_block_layers=self.fc_block_layers,
            fc_hidden_units=self.fc_hidden_units,
            block_sharing=self.block_sharing,
            dropout=self.dropout
        )

        # create list stacks with at least one block
        self.stacks = nn.ModuleList([
            NBeatsStack(
                input_size=self.input_size,
                output_size=self.output_size,
                num_blocks=self.num_blocks,
                fc_block_layers=self.fc_block_layers,
                fc_hidden_units=self.fc_hidden_units,
                block_sharing=self.block_sharing,
                dropout=self.dropout
            ) for _ in range(num_stacks)
        ])

    def forward(self, res,x_dec=None):
        seq_last = res[:, -1:, :].detach()
        res = res - seq_last
        residuals = res
        self.forecast = 0
        for i, stack in enumerate(self.stacks):
            residuals, stack_forecast = stack(residuals)
            self.forecast = stack_forecast + stack_forecast
        self.forecast = self.forecast + seq_last

        return self.forecast

    # def model(self):
    #     # get residuals from first stack
    #     stack_input = torch.zeros((1, self.input_size))  # create dummy input
    #     residuals, _ = self.nbeats_stack(stack_input)
    #     return residuals, self.forward(residuals)