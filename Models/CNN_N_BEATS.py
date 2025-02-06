import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.N_BEATS_M import NBeats_M

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()                
        self.kernel_size = 3
        self.dense_cnn_units_layer = args.forecast_horizon
        self.cnn_dropout = args.dropout
        self.batch_size = args.batch_size
        self.num_series = args.num_series
        self.time_steps = args.past_history
        self.f_horizon = args.forecast_horizon
        self.filters = 8 
        self.pool_size = 2
        self.dilation=1
            
        self.conv1d = nn.Conv1d(in_channels=self.num_series, out_channels=self.filters, kernel_size=self.kernel_size, padding='same', dilation=self.dilation) 
        self.conv2d = nn.Conv1d(in_channels=self.filters, out_channels=self.num_series, kernel_size=self.kernel_size, padding='same', dilation=self.dilation)  
        self.maxPool1 = nn.MaxPool1d(kernel_size=self.pool_size)
        self.maxPool2 = nn.MaxPool1d(kernel_size=self.pool_size)
        self.dense_layer = nn.Linear( round(self.time_steps/4) ,self.f_horizon)
        self.dropout = nn.Dropout(self.cnn_dropout)
        self.conv_1 = nn.Conv1d(in_channels=self.time_steps, out_channels=self.filters, kernel_size=self.kernel_size, padding='same', dilation=self.dilation) 
        self.linear_1  = nn.Linear(self.filters,self.f_horizon)
        self.skip_layer = nn.Linear(self.time_steps , self.f_horizon)
        self.m_n_beats = NBeats_M(
            input_size=(self.f_horizon, args.num_series),
            output_size=args.forecast_horizon,
            num_stacks=args.num_stacks,
            num_blocks=args.num_blocks,
            fc_block_layers=args.fc_hidden_layers,
            fc_hidden_units=args.fc_hidden_units,
            block_sharing=args.block_sharing,
            dropout=args.dropout,
            #device=args.device
        )


    def forward(self, inputs, dec_inp):
        #inputs [B,past_history,num_series]
        x = inputs          
        seq_last = x[:, -1:, :].detach() # [B,1,num_series]  #
        x = x - seq_last
        x = x.permute(0, 2, 1) #to (batch_size,num_series,time_steps)   
        x = self.conv_1(x.permute(0,2,1)).permute(0,2,1) #to (batch_size,filters(num_series),time_steps)                 
        x  = self.linear_1(x) #to(batch_size, num_series,f_horizon)
        skip_output = self.skip_layer(inputs.permute(0,2,1)).permute(0,2,1)
        x = x.permute(0,2,1) #+ skip_output
        
        # x =self.m_n_beats(x)
        # print("x vant 3" , x.shape)  
        #x = F.relu(self.conv1d(x))    #to (batch_size,filters(num_series),time_steps)      
        #x = self.maxPool1(x)  #to (batch_size,filters(num_series),time_steps_max_p_1)
        #x = F.relu(self.conv2d(x))    #to (batch_size,num_series,time_steps_max_p_1)
        #x = self.maxPool2(x)  #to (batch_size,num_series,time_steps_max_p_2)
        #x = self.dense_layer(x)     
        #x = x.permute(0,2,1)
        x = x + seq_last
        return x

# class NBeatsBlock(nn.Module):
#     def __init__(self, input_size: int, output_size: int, theta_size: int, fc_block_layers: int, fc_hidden_units: int, dropout: float):
#         super(NBeatsBlock, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.theta_size = theta_size
#         self.fc_hidden_units = nn.ModuleList([nn.Linear(fc_hidden_units, fc_hidden_units) for _ in range(fc_block_layers)])
#         self.theta_layer = nn.Linear(fc_hidden_units, self.theta_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         for layer in self.fc_hidden_units:
#             x = F.relu(layer(x))
#             x = self.dropout(x)
#         theta = self.theta_layer(x)
#         backcast, forecast = theta[:, :self.input_size], theta[:, -self.output_size:]
#         return backcast, forecast

# class NBeatsStack(nn.Module):
#     def __init__(self, input_size: int, output_size: int, num_blocks: int, fc_block_layers: int, fc_hidden_units: int, block_sharing: bool, dropout: float):
#         super(NBeatsStack, self).__init__()
#         self.blocks = nn.ModuleList([NBeatsBlock(input_size=input_size, output_size=output_size, theta_size=input_size + output_size, fc_block_layers=fc_block_layers, fc_hidden_units=fc_hidden_units, dropout=dropout)])
#         if not block_sharing:
#             self.blocks.extend([NBeatsBlock(input_size=input_size, output_size=output_size, theta_size=input_size + output_size, fc_block_layers=fc_block_layers, fc_hidden_units=fc_hidden_units, dropout=dropout) for _ in range(1, num_blocks)])

#     def forward(self, residuals):
#         forecast = torch.zeros_like(residuals)
#         for block in self.blocks:
#             backcast, forecast_block = block(residuals)
#             forecast += forecast_block
#             residuals -= backcast
#         return residuals, forecast

# class NBeats(nn.Module):
#     def __init__(self, input_size: int, output_size: int, num_stacks: int, num_blocks: int, fc_block_layers: int, fc_hidden_units: int, block_sharing: bool, dropout: float):
#         super(NBeats, self).__init__()
#         self.nbeats_stack = NBeatsStack(input_size=input_size, output_size=output_size, num_blocks=num_blocks, fc_block_layers=fc_block_layers, fc_hidden_units=fc_hidden_units, block_sharing=block_sharing, dropout=dropout)
#         self.stacks = nn.ModuleList([NBeatsStack(input_size=input_size, output_size=output_size, num_blocks=num_blocks, fc_block_layers=fc_block_layers, fc_hidden_units=fc_hidden_units, block_sharing=block_sharing, dropout=dropout) for _ in range(num_stacks-1)])

#     def forward(self, residuals):
#         forecast = torch.zeros_like(residuals)
#         residuals, stack_forecast = self.nbeats_stack(residuals)
#         forecast += stack_forecast
#         for stack in self.stacks:
#             residuals, stack_forecast = stack(residuals)
#             forecast += stack_forecast
#         return forecast



# class CNN_N_BEATS(nn.Module):
#     def __init__(self, args):
#         super(CNN_N_BEATS, self).__init__()

#         self.filters = 64      # tendance incrémentale
#         self.kernel_size = 2
#         self.cnn_input_shape = (args.past_history,args.num_series)
#         self.pool_size = 2
#         self.dense_cnn_units_layer = args.forecast_horizon
#         self.cnn_dropout = 0.2
#         #n_beats hyperparameters
#         self.n_beats_input = args.forecast_horizon
#         self.n_beats_output = args.forecast_horizon
#         self.num_stacks = 10  
#         self.num_blocks = 1
#         self.fc_block_layers = 4
#         self.fc_hidden_units = 64
#         self.block_sharing = False
#         self.n_beats_dropout = 0.2

#         # CNN parameters
#         self.cnn_trans = CNN(
#             cnn_input_shape=self.cnn_input_shape,
#             filters=self.filters,
#             kernel_size=self.kernel_size,
#             pool_size=self.pool_size,
#             dense_cnn_units_layer=self.dense_cnn_units_layer,
#             dropout=self.cnn_dropout
#         )

#         # NBeats parameters
#         self.n_beats = NBeats(
#             input_size=self.n_beats_input,
#             output_size=self.n_beats_output,
#             num_stacks=self.num_stacks,
#             num_blocks=self.num_blocks,
#             fc_block_layers=self.fc_block_layers,
#             fc_hidden_units=self.fc_hidden_units,
#             block_sharing=self.block_sharing,
#             dropout=self.n_beats_dropout
#         )
#     def forward(self, x,dec_inp ):
#         seq_last = x[:, -1:, :]  # Take the last sequence element
#         x = x - seq_last  # Normalize by subtracting last element

#         x = self.cnn_trans(x)
#         x = self.n_beats(x)

#         x = x.unsqueeze(2)  # Reshape to match dimensions
#         x = x + seq_last
#         x = x.squeeze(2)  # Remove the extra dimension

#         return x
