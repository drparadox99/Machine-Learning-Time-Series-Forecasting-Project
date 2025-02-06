import import_ipynb
import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from  sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#from DataPreprocessing import DataPreprocessing
#import metrics
#%matplotlib inline
#from plot import _plot
#import tensorflow_addons as tfa
import os
from MultiHeadAttention import MultiHeadAttention_

print(tf.__version__)


#------ChatGPT

#Encoder Class

class Encoder(tf.keras.layers.Layer):
    def __init__( self, num_heads: int, d_model: int, ff_dim: int, trans_input_shape : tuple , encoder_dropout: int, conv_1D_kernel_size=1, ** kwargs):
        super().__init__(**kwargs)
        #self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.encoder_dropout = encoder_dropout
        self.conv_1D_kernel_size = conv_1D_kernel_size
        #Multi-attention block
        #self.multi_attention_block = tf.keras.layers.MultiHeadAttention( key_dim=d_model, num_heads=num_heads, dropout=encoder_dropout )
        self.multi_attention_block_ = MultiHeadAttention_(num_heads=num_heads,d_model=d_model,dropout=encoder_dropout)

        self.dropout =  tf.keras.layers.Dropout(encoder_dropout)
        self.normalization_layer =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #FF Layers  # basic a linear layer
        self.conv_1D_layer_1 =  tf.keras.layers.Conv1D(filters=self.ff_dim, kernel_size=self.conv_1D_kernel_size, activation="relu")
        self.ff_dropout =  tf.keras.layers.Dropout(encoder_dropout)
        self.conv_1D_layer_2 =  tf.keras.layers.Conv1D(filters=trans_input_shape[1], kernel_size=self.conv_1D_kernel_size, activation="relu")
        self.ff_layer_normalization =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
    


    def call(self, inputs,training=False):  # the call method is what runs when the layer is called        
            
        #x = self.multi_attention_block(inputs,inputs)
        x,o = self.multi_attention_block_(inputs)  
        x = self.dropout(x)
        x = self.normalization_layer(x)
        res = x + inputs #residual connection
        #FF Layers
        x = self.conv_1D_layer_1(res)
        x = self.ff_dropout(x)
        x = self.conv_1D_layer_2(x)
        x = self.ff_layer_normalization(x)
        return x + res


#Transformers Class 

class Transformers(tf.keras.Model):
    def __init__(self,
        trans_input_shape:tuple,    
        transformers_output:int,
        num_encoders: int,
        num_heads: int,
        d_model: int,
        ff_dim: int,
        ff_layers_hidden_units: int,
        encoder_dropout: int,
        ff_layers_dropout: int,
        conv_1D_kernel_size:int,
         ** kwargs):
        super().__init__(**kwargs)


        self.trans_input_shape = trans_input_shape 
        self.transformers_output = transformers_output  # forecast horizon
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.ff_layers_hidden_units = ff_layers_hidden_units # list hidden layers' units
        self.encoder_dropout = encoder_dropout
        self.ff_layers_dropout = ff_layers_dropout
        self.conv_1D_kernel_size = conv_1D_kernel_size  # default value = 1 

        self.encoders = [Encoder(
            num_heads=self.num_heads,
            d_model=self.d_model,
            ff_dim=self.ff_dim ,
            trans_input_shape=trans_input_shape,
            encoder_dropout=self.encoder_dropout,
            conv_1D_kernel_size=conv_1D_kernel_size) for _ in range(self.num_encoders)]

        self.globalAveragePooling1D_layer =  tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")
        self.ff_layers  = [ tf.keras.layers.Dense(unit)  for unit in self.ff_layers_hidden_units]
        self.ff_dropout =  tf.keras.layers.Dropout(self.ff_layers_dropout)

        #transformers' output 
        self.transformers_output_layer  =  tf.keras.layers.Dense(self.transformers_output)
        #transformers' input layer
        self.transformers_input_layer = tf.keras.Input(shape=self.trans_input_shape)


    def call(self, inputs,training=False):  # the call method is what runs when the layer is called
        x = inputs
        for encoder in self.encoders:
          
            x  = encoder(x)
        x = self.globalAveragePooling1D_layer(x)
        for layer in self.ff_layers:
            x = layer(x)
            x  = self.ff_dropout(x)
        output =  self.transformers_output_layer(x)
        return output 

    def model(self):
        return tf.keras.Model(inputs=self.transformers_input_layer, outputs=self.call(self.transformers_input_layer), name="Transformers")




#NBeats Block Class

class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        theta_size: int,
        fc_block_layers: int,
        fc_hidden_units: int,
        dropout: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.theta_size = theta_size
        self.fc_layers = []
        self.dropout = dropout

        # for i in range(fc_block_layers):
        #    self.fc_layers.append(tf.keras.layers.Dense(
        #        fc_hidden_units, activation=tf.nn.relu))

        self.fc_hidden_units = [tf.keras.layers.Dense(
            fc_hidden_units, activation="relu") for _ in range(fc_block_layers)]

        # self.intermediate_layer = tf.keras.layers.Dense(2000, activation="relu")

        self.theta_layer = tf.keras.layers.Dense(
            self.theta_size, activation="linear", name=f"ThetaLayer")

        # self.forecast = tf.keras.layers.Dense(output_size, activation="linear", name=f"forecast_block_layer")
        # self.backcast = tf.keras.layers.Dense(input_size, activation="linear", name=f"backcast_block_layer")

        self.dropout = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.fc_hidden_units:  # pass inputs through each hidden layer
            x = layer(x)
            x = self.dropout(x)

        # x = self.intermediate_layer(x)

        theta = self.theta_layer(x)
        # Output the backcast and forecast from theta
        backcast, forecast = theta[:,:self.input_size], theta[:, -self.output_size:]
        return backcast, forecast
        '''         
                inputs = x
                for layer in self.fc_layers:
                    x = layer(x)
                    # = self.dropout(x)
                backcast = tf.keras.activations.relu(inputs - self.backcast(x))
                return backcast, self.forecast(x)
        '''



#NBeats Stack Class

class NBeatsStack(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_blocks: int,
        fc_block_layers: int,
        fc_hidden_units: int,
        block_sharing: bool,
        dropout: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.blocks = [NBeatsBlock(input_size=input_size, output_size=output_size, theta_size=input_size + output_size,
                                    fc_block_layers=fc_block_layers, fc_hidden_units=fc_hidden_units, dropout=dropout, name=f"NBeatsBlockInitial")]
        self.output_size = output_size

        for i in range(1, num_blocks):
            if block_sharing:
                self.blocks.append(self.blocks[0])
            else:
                self.blocks.append(NBeatsBlock(input_size=input_size, output_size=output_size, theta_size=input_size + output_size,
                                                fc_block_layers=fc_block_layers, fc_hidden_units=fc_hidden_units, dropout=dropout, name=f"NBeatsBlock_{i}"))

    def call(self, residuals, training=False):
        forecast = tf.convert_to_tensor(0.0)
        # residuals = 0.0
        for _, block in enumerate(self.blocks):
            # print("NNBeatsBlock Call")
            backcast, forecast_block = block(residuals)
            # forecast = tf.keras.layers.add([forecast, forecast_block])
            forecast = forecast + forecast_block
            residuals = residuals - backcast  # calculate resisual

            # residuals = tf.keras.layers.subtract([residuals, backcast],name=f"subtract_{index}")  #calculate resisual
        return residuals, forecast  # * level


#NBeats Class 
class NBeats(tf.keras.Model):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_stacks: int,
        num_blocks: int,
        fc_block_layers: int,
        fc_hidden_units: int,
        block_sharing: bool,
        dropout: float
    ):
        super().__init__()
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
            dropout=self.dropout,
            name="InitialStack"
        )

        # create list stacks with at least one block
        self.stacks = [NBeatsStack(
            input_size=self.input_size,
            output_size=self.output_size,
            num_blocks=self.num_blocks,
            fc_block_layers=self.fc_block_layers,
            fc_hidden_units=self.fc_hidden_units,
            block_sharing=self.block_sharing,
            dropout=self.dropout,
            name=f"Stack_{i}"
        ) for i in range(num_stacks-1)]

 

    def call(self, res, training=False):
        residuals = res
        self.forecast = 0.0
        #forecast = tf.convert_to_tensor(0.0)
        for i, _ in enumerate(range(len(self.stacks))):
            # print("NBeatsStacks Call")
            residuals, stack_forecast = self.stacks[i](residuals)
            #self.forecast = tf.keras.layers.add([self.forecast, stack_forecast])
            self.forecast = self.forecast + stack_forecast
        return self.forecast

    def model(self):
        # get residuals from first stack
        self.stack_input = tf.keras.layers.Input(shape=(self.input_size), name="stack_input")
        residuals, self.forecast = self.nbeats_stack(self.stack_input)
        return tf.keras.Model(inputs=self.stack_input, outputs=self.call(residuals), name="K_NBEATS")


            


class TRANS_BEATS_Uni(tf.keras.Model):
    def __init__(self,
        trans_input_shape:tuple,    
        transformers_output:int,
        num_encoders: int,
        num_heads: int,
        d_model: int,
        ff_dim: int,
        ff_layers_hidden_units: int,
        encoder_dropout: int,
        ff_layers_dropout: int,
        conv_1D_kernel_size:int,

        num_stacks: int,  
        num_blocks: int,
        fc_block_layers: int,
        fc_hidden_units: int,
        block_sharing: bool,
        n_beats_dropout: float,
         ** kwargs):
        super().__init__(**kwargs)

        #transformers parameters
        self.trans_inpu_shape = trans_input_shape
        self.transformers_output = transformers_output
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.d_model = d_model 
        self.ff_dim = ff_dim
        self.ff_layers_hidden_units = ff_layers_hidden_units
        self.encoder_dropout = encoder_dropout 
        self.ff_layers_dropout = ff_layers_dropout 
        self.conv_1D_kernel_size = conv_1D_kernel_size

        #NBeats parameters
        self.num_stacks = num_stacks 
        self.num_blocks = num_blocks 
        self.fc_block_layers = fc_block_layers
        self.ff_hidden_units = fc_hidden_units 
        self.block_sharing = block_sharing
        self.n_beats_dropout = n_beats_dropout 

        self.transformers = Transformers(
            trans_input_shape=trans_input_shape,            
            transformers_output= transformers_output,
            num_encoders= num_encoders,
            num_heads= num_heads,
            d_model= d_model,
            ff_dim= ff_dim,
            ff_layers_hidden_units= ff_layers_hidden_units,
            encoder_dropout= encoder_dropout,
            ff_layers_dropout= ff_layers_dropout,
            conv_1D_kernel_size=conv_1D_kernel_size
        )
        self.n_beats = NBeats(
        input_size=transformers_output,
        output_size= transformers_output,
        num_stacks= num_stacks,
        num_blocks= num_blocks,
        fc_block_layers= fc_block_layers,
        fc_hidden_units= fc_hidden_units,
        block_sharing= block_sharing,
        dropout= n_beats_dropout)



    def call(self): #used for transformers uni and transbeats uni
        x = self.transformers(self.transformers.transformers_input_layer)
        x  = self.n_beats(x)
        return x 

    def model(self):
        print("dans le premier !")
        return tf.keras.Model(inputs=self.transformers.transformers_input_layer, outputs=self.call(), name="TRANS_BEATS_Uni")


#input_tensor = tf.keras.Input(shape=(328, 1))
#toy_encodings = tf.random.normal(tf.shape(input_tensor))