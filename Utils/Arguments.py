
class Args:
    def __init__(self):
        self.device='cpu'
        self.train_size=0.7
        self.test_size=0.2
        self.val_size=0.1
        self.training_duplicates=1
        self.testing_duplicates=0
        self.batch_size=32  #512
        self.n_epochs=60
        #let past_history=$forecast_horizon*3  #optimal #188
        self.lr=0.0001 #0.001 #0.002300000051036477  #0.0001  #0.01 ##0.0001
         # Shuffle training data
        self.shuffle=True               #boolean
        self.selected_model="MLP"
        self.save_optimal_model=False    #boolean
        #model hyperparameters
        #input_size=($past_history,$num_series),
        self.kernel_size=25  #optimal
        self.num_stacks=1
        self.num_blocks=1
        self.fc_hidden_layers=1
        self.fc_hidden_units=32
        self.block_sharing=False  #boolean
        self.dropout=0.0
        self.hidden_layers_dropout=0.0
        self.dataset_name='exchange_rate'
        self.num_series=8
        self.data_file_path="Datasets/"+self.dataset_name+".csv"
        self.forecast_horizon= 720   #48 96 192 336 720
        self.past_history= self.forecast_horizon #self.forecast_horizon * 3
        self.exec_name="dataset_name_fh_"+str(self.forecast_horizon)+"_ph_"+str(self.past_history)
        self.save_model_PATH="Checkpoint_file/pt_models/"+self.selected_model+"_"+self.exec_name

        #formers
        self.label_len=48
        self.output_attention=False
        self.embed_type=0  #[0,1]
        self.embed="fixed" # ["timeF", "fixed", "learned"]
        self.freq = "h"  # ["s", "t", "h", "d","b", "w", "m"]
        self.activation='gelu'
        self.factor=1
        self.d_model=128
        self.n_heads=1
        self.d_ff=1024
        self.d_layers=1
        self.e_layers=1
        self.distil=False #informer

    def print_args(self):
        print(
            "Args:\n device: {0}\n train_size:{1}\n test_size:{2}\n val_size:{3}\n training_duplicates:{4}\n testing_duplicates:{5}\n batch_size:{6}\n n_epochs:{7}\n lr:{8}\n shuffle:{9}\n selected_model:{10}\n save_optimal_model:{11}\n kernel_size:{12}\n num_stacks:{13}\n num_blocks:{14}\n fc_hidden_layers:{15}\n fc_hidden_units:{16}\n block_sharing:{17}\n dropout:{18}\n hidden_layers_dropout:{19}\n dataset_name:{20}\n num_series:{21}\n data_file_path:{22}\n forecast_horizon:{23}\n past_history:{24}\n exec_name:{25}\n save_model_PATH:{26} \n label_len:{27} \n output_attention:{28} \n embed_type:{29} \n embed:{30} \n freq:{31} \n activation:{32} \n factor:{33} \n d_model:{34} \n n_heads:{35} \n d_ff:{36} \n d_layers:{37} \n e_layers:{38} \n distil:{39}"
            .format(
          self.device,
            self.train_size,
            self.test_size,
            self.val_size,
            self.training_duplicates,
            self.testing_duplicates,
            self.batch_size,
            self.n_epochs,
            self.lr,
            self.shuffle,
            self.selected_model,
            self.save_optimal_model,
            self.kernel_size,
            self.num_stacks,
            self.num_blocks,
            self.fc_hidden_layers,
            self.fc_hidden_units,
            self.block_sharing,
            self.dropout,
            self.hidden_layers_dropout,
            self.dataset_name,
            self.num_series,
            self.data_file_path,
            self.forecast_horizon,
            self.past_history,
            self.exec_name,
            self.save_model_PATH,
            self.label_len,
            self.output_attention,
            self.embed_type,
            self.embed,
            self.freq,
            self.activation,
            self.factor,
            self.d_model,
            self.n_heads,
            self.d_ff,
            self.d_layers,
            self.e_layers,
            self.distil
            )
    )

args =  Args()


