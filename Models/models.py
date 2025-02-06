
from Models.LTSF_Linear import DLinearModel,Simple_Linear,N_Linear
from Models.Final_Models import DC_BEATS,Deep_Blocks
from Models.N_BEATS import NBeatsNet
from Models.N_Hits import _Block
from Models.Transformer import Transformers #,TransformerEncoderModel
from Models.GMLP import GMLP
from Models.GMLP_2 import GMLP_2

from Models.Models_DeadEnd.TS_Mixer import TS_Mixer
from Models.SegRNN import Model as segRNN
from Models.Models_DeadEnd.PatchTST_.PatchTST import Model as Patch_tst
from Models.Models_DeadEnd.MSD_Mixer import MSDMixer
#from Models.Stacked_N_Linear import Stack_N_Linear
from Models.BEATS_CELL import BEATS_CELL
from Models.iTrans.iTransformer import Model as iTransModel
from Models.Formers.Autoformer.Autoformer import Model as Autoformer
from Models.Formers.Informer.Informer import Model as Informer
from Models.Formers.VanillaTransformer.VanillaTransformer import Model as VanillaTransformer
from Models.Formers.FEDformer.FEDformer import Model as FEDformer
from Models.Mamba.Mamba import Mamba, MambaConfig
from Models.Formers.VanillaTransformer.Informer_Transformer_Decoder import Model as Informer_Transformer_Decoder
from Models.BEATS_CELL_Trans_Dec import BEATS_CELL_Trans_Dec
from Models.Formers.MinusFormer.Minusformer import Model as Minusformer
from Models.MOE import N_Linear_MOE
from Models.RLinear import Model as RLinear
from Models.Formers.VanillaTransformer.NS_Transformer import Model as NS_Transformer
from Models.RMLP import Model as RMLP

# from Models.minGRU import Model as minGRU
from Models.N_BEATS_M import NBeats_M
from Models.CNN_N_BEATS import CNN

# from Models.Formers.VanillaTransformer.MambaFormer import Model as MambaFormer

def get_model(args):
    selected_model = ""
    print("Args.selected_model: ", args.selected_model)

    if args.selected_model == 'Deep_Blocks':
        selected_model = Deep_Blocks(
                            input_size=(args.past_history,args.num_series),
                            forecast_horizon=args.forecast_horizon,
                            num_blocks=args.num_blocks,
                            num_fc_layers= args.fc_hidden_layers,
                            expansion_dim= args.fc_hidden_units,
                            dropout= args.dropout,
                            device=args.device
                            )


    if args.selected_model == 'N_BEATS':
        selected_model = NBeatsNet(
                  device=args.device,
                  stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK],
                  forecast_length=args.forecast_horizon,
                  nb_blocks_per_stack=2,
                  thetas_dim=(4, 8), #4,8
                  share_weights_in_stack=False,
                  backcast_length=args.past_history,
                  hidden_layer_units=256,
                  #nb_harmonics=None
                  )

    if args.selected_model == 'GMLP':
        selected_model = GMLP(input_dim=(args.past_history,args.num_series),forecast_horizon=args.forecast_horizon)

    if args.selected_model == 'GMLP_2':
        selected_model = GMLP_2(args.past_history,args.num_series,args.forecast_horizon,2)

    if args.selected_model == 'TS_Mixer':
        selected_model = TS_Mixer(args)

    if args.selected_model == "Patch_tst":
        selected_model = Patch_tst(args)

    if args.selected_model == "msd_mixer":
        selected_model = MSDMixer(
            in_len = args.past_history,
            out_len =  args.forecast_horizon,
            in_chn = args.num_series,
            ex_chn = 0 ,
            out_chn = args.num_series,
            patch_sizes =  [28, 14, 7, 2, 1] ,
            hid_len = 32,
            hid_chn = 256,
            hid_pch = 32,
            hid_pred = 32,
            norm = None,
            last_norm = True,
            activ = "gelu",
            drop= 0.0,
        )

    #if args.selected_model == "Stack_N_Linear":
    #    selected_model = Stack_N_Linear((args.past_history,args.num_series),args.forecast_horizon , False,2,128,0.8)

    #if args.selected_model == "Stack_N_Linear":
    #   selected_model = Stack_N_Linear(args.past_history,args.forecast_horizon,args.num_series,1,-1,0.8)

    if args.selected_model == "iTransformer":
        selected_model = iTransModel(args)

    # params = {
    #         "seq_len" : args.past_history ,
    #         "label_len" : 48,
    #         "pred_len" : args.forecast_horizon,
    #         "output_attention" : False ,
    #         "kernel_size" : 25 ,
    #         "dropout" : 0.1,
    #         "embed_type" : 0 , #[0, 1,2,3,4] # 0: pos embedding 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
    #         "embed" : "fixed",#["timeF", "fixed", "learned"],
    #         "freq" : "h", #["s", "t", "h", "d","b", "w", "m"],
    #         "dropout" : 0.0,
    #         "activation" : 'gelu' ,
    #         "factor" : 1 ,
    #         "d_model" : 128 ,
    #         "n_heads" : 1,
    #         "d_ff" : 32  ,
    #         "d_layers"  : 1,
    #         "e_layers" : 1 ,
    #         "enc_in" : args.num_series ,  #num_series
    #         "dec_in" : args.num_series , #num_series
    #         "c_out" :  args.num_series ,   #num_series
    #         "device" : args.device
    # }

    if args.selected_model == 'Transformer':
        selected_model = Transformers(
            lookback_w=args.past_history,
            num_series=args.num_series,
            forecast_h=args.forecast_horizon,
            num_encoders=args.e_layers,
            num_heads=args.n_heads,
            #head_size=int(args.past_history / 4),
            d_model=int(args.past_history * 2),
            ff_dim=args.d_ff,
            ff_layers_hidden_units=[args.fc_hidden_units],
            encoder_dropout=args.dropout,
            ff_layers_dropout=args.dropout
        )

    if args.selected_model == "BEATS_CELL":
        selected_model = BEATS_CELL(args)

    if args.selected_model == 'Autoformer':
        selected_model = Autoformer(args)

    if args.selected_model == 'Informer':
        selected_model = Informer(args)

    if args.selected_model == 'VanillaTransformer':
        selected_model = VanillaTransformer(args)

    if args.selected_model == 'Informer_Transformer_Decoder':
        selected_model = Informer_Transformer_Decoder(args)

    if args.selected_model == 'FEDformer':
        class Configs(object):
            ab = 0
            modes = 2
            mode_select = 'random'
            # version = 'Fourier'
            version = 'Wavelets'
            moving_avg = [13]
            L = 1
            base = 'legendre'
            cross_activation = 'tanh'
            past_history = args.past_history
            label_len = 48
            forecast_horizon = args.forecast_horizon
            output_attention = False
            num_series = args.num_series
            num_series = args.num_series
            d_model = 16
            embed = 'fixed'
            embed_type = 0
            dropout = 0.0
            freq = 'h'
            factor = 1
            n_heads = 1
            d_ff = 32
            e_layers = 1
            d_layers = 1
            num_series = args.num_series
            activation = 'gelu'
        configs = Configs()
        selected_model = FEDformer(configs)

    if args.selected_model == 'segRNN':
        selected_model = segRNN(args)


    if args.selected_model == 'DLinearModel':
        selected_model = DLinearModel(
            kernel_size=args.kernel_size,
            seq_len=args.past_history,
            pred_len=args.forecast_horizon,
            individual=False,
            num_series=args.num_series,
            device=args.device
        )
    if args.selected_model == 'Simple_Linear':
        selected_model = Simple_Linear(input_size=(args.past_history, args.num_series),
                                       forecasting_horizon=args.forecast_horizon,device=args.device)

    if args.selected_model == 'N_Linear':
        selected_model = N_Linear(args)


    if args.selected_model == "Mamba":
        config = MambaConfig(d_model=128,lookback_w=args.past_history,num_series=args.num_series,forecast_h=args.forecast_horizon,n_layers=1)
        selected_model = Mamba(config)


    if args.selected_model == "BEATS_CELL_Trans_Dec":
        selected_model = BEATS_CELL_Trans_Dec(args,lookback_w=args.past_history,forecast_h=args.forecast_horizon,num_series=args.num_series,hidden_units=args.d_model,num_cells=args.e_layers,cell_dropout=args.dropout,dropout=args.dropout)


    if args.selected_model == 'DC_BEATS':
        selected_model = DC_BEATS(
            input_size=(args.past_history, args.num_series),
            output_size=args.forecast_horizon,
            kernel_size=args.kernel_size,
            num_stacks=args.num_stacks,
            num_blocks=args.num_blocks,  # essayer 4
            fc_hidden_layers=args.fc_hidden_layers,
            fc_hidden_units=args.fc_hidden_units,
            block_sharing=args.block_sharing,
            dropout=args.dropout,
            device=args.device
        )


    if args.selected_model == 'NBeats_M':
        selected_model = NBeats_M(
            input_size=(args.past_history, args.num_series),
            output_size=args.forecast_horizon,
            num_stacks=args.num_stacks,
            num_blocks=args.num_blocks,
            fc_block_layers=args.fc_hidden_layers,
            fc_hidden_units=args.fc_hidden_units,
            block_sharing=args.block_sharing,
            dropout=args.dropout,
            #device=args.device
        )

    if args.selected_model == 'Minusformer':
        selected_model = Minusformer(args)

    if args.selected_model == 'N_Linear_MOE':
        selected_model = N_Linear_MOE(args)

    if args.selected_model == 'RLinear':
        selected_model = RLinear(args)

    if args.selected_model == 'NS_Transformer':
        selected_model = NS_Transformer(args)

    if args.selected_model == "RMLP":
        selected_model = RMLP(args)

    if args.selected_model == "CNN_N_BEATS":
        selected_model = CNN(args)
    
    # if args.selected_model == "minGRU":
    #     selected_model = minGRU(args)

    # if args.selected_model == 'N_HITS':
    #     selected_model = _Block(
    #         input_chunk_length=args.past_history,
    #         output_chunk_length=args.forecast_horizon,
    #         num_layers = args.fc_hidden_layers,
    #         layer_width = args.fc_hidden_layers,
    #         nr_params = 1,
    #         pooling_kernel_size =args.kernel_size ,
    #         n_freq_downsample = 1 ,
    #         batch_norm = False ,
    #         dropout = 0.2 ,
    #         MaxPool1d = 2
    #     )
    # selected_model = _NHiTSModule(
    #             input_dim=args.past_history,
    #             output_dim=args.forecast_horizon,
    #             nr_params=1,
    #             num_stacks=args.num_stacks,
    #             num_blocks=args.num_blocks,
    #             num_layers=args.fc_hidden_layers,
    #             layer_widths=[args.fc_hidden_units],
    #             pooling_kernel_sizes=[args.kernel_size],
    #             n_freq_downsample=[1],
    #             batch_norm=False,
    #             dropout=0.2,
    #             activation='relu',
    #             MaxPool1d=2)


    return selected_model