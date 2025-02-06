#!/bin/sh
#SBATCH --job-name=job
#SBATCH --partition=quadgpu
#SBATCH --gres=gpu:2  
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kend4r@hotmail.com     # e-mail notification
#SBATCH --output=job_revue%j.out          # if --error is absent, includes also the errors
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10


echo "------------------------------------------------------------------------------"
echo "hostname                     =   $(hostname)"
echo "SLURM_JOB_NAME               =   $SLURM_JOB_NAME"
echo "SLURM_SUBMIT_DIR             =   $SLURM_SUBMIT_DIR"
echo "SLURM_JOBID                  =   $SLURM_JOBID"
echo "SLURM_JOB_ID                 =   $SLURM_JOB_ID"
echo "SLURM_NODELIST               =   $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST           =   $SLURM_JOB_NODELIST"
echo "SLURM_TASKS_PER_NODE         =   $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE       =   $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_TOPOLOGY_ADDR_PATTERN  =   $SLURM_TOPOLOGY_ADDR_PATTERN"
echo "SLURM_TOPOLOGY_ADDR          =   $SLURM_TOPOLOGY_ADDR"
echo "SLURM_CPUS_ON_NODE           =   $SLURM_CPUS_ON_NODE"
echo "SLURM_NNODES                 =   $SLURM_NNODES"
echo "SLURM_JOB_NUM_NODES          =   $SLURM_JOB_NUM_NODES"
echo "SLURMD_NODENAME              =   $SLURMD_NODENAME"
echo "SLURM_NTASKS                 =   $SLURM_NTASKS"
echo "SLURM_NPROCS                 =   $SLURM_NPROCS"
echo "SLURM_MEM_PER_NODE           =   $SLURM_MEM_PER_NODE"
echo "SLURM_PRIO_PROCESS           =   $SLURM_PRIO_PROCESS"
echo "------------------------------------------------------------------------------"

# USER Commands

# special commands for openmpi/intel
#module load openmpi/intel-opa/gcc/64
#module load openmpi/gcc/64/4.1.2
#module load gcc/11.2.0

time hostname

#dont forget to add the save_optimal_model parameter in the 4 loop later on

device='cuda' #['cpu','cuda']
#training parameters
train_size=0.7
test_size=0.2
val_size=0.1
training_duplicates=1
testing_duplicates=0
#forecast_horizon=96   #96 192 336 720
batch_size=32  #512
n_epochs=60
#let past_history=$forecast_horizon*3  #optimal #188
lr=0.001  #0.001 #0.002300000051036477  #0.0001  #0.01 ##0.0001
 # Shuffle training data
shuffle=True               #boolean
#selected_model="N_Linear" #[VanillaTransformer,Informer, Autoformer, Patch_tst, RLinear, RMLP, N_Linear, Simple_Linear, DLinearModel,segRNN,N_Linear_MOE]
save_optimal_model=False    #boolean --save_optimal_model \
#model hyperparameters
#input_size=($past_history,$num_series),
kernel_size=25  #optimal
num_stacks=1
num_blocks=1
fc_hidden_layers=1
fc_hidden_units=128
block_sharing=False       #boolean
dropout=0.0
hidden_layers_dropout=0.0
dataset_name='ETTm1'
num_series=7
dataset_num_series=$num_series
data_file_path="Datasets/${dataset_name}.csv"
#exec_name="${dataset_name}_fh_${forecast_horizon}_ph_${past_history}"
#save_model_PATH="Checkpoint_file/pt_models/${selected_model}_${exec_name}"
label_len=48
output_attention=False       #boolean
embed_type=0  #[0,1]
embed='fixed' # ["timeF", "fixed", "learned"]
freq='h'  # ["s", "t", "h", "d","b", "w", "m"]
activation='gelu'
factor=1
d_model=128
n_heads=1
d_ff=32
d_layers=1
e_layers=1
distil=False #informer
#approach="global" #[local,global,clustering]
exec_context="exec_context_1"
exec_iterations=1
custom_val_split=0
invs_dataset_aug=0


for selected_model in 'VanillaTransformer' 'Informer' 'Autoformer' 'Patch_tst' 'RLinear' 'RMLP' 'N_Linear' 'Simple_Linear' 'DLinearModel' 'segRNN' 'N_Linear_MOE'
do
  for approach in 'local' 'global' 'clustering'
    do
      echo "------------------------------------Forecasting approach ----------------------------------->:"$approach
      for forecast_horizon in 48 96 192 336 720
      do
          let past_history=$(($forecast_horizon))
          exec_name="${dataset_name}_fh_${forecast_horizon}_ph_${past_history}"
          save_model_PATH="Checkpoint_file/pt_models/${selected_model}_${exec_name}"
          if [ $forecast_horizon -le 48 ]
          then
          label_len=16
          else
          label_len=48
          fi
          python Pytorch_Main.py --device $device \
                              --train_size $train_size \
                              --test_size $test_size \
                              --val_size $val_size \
                              --training_duplicates $training_duplicates \
                              --testing_duplicates $testing_duplicates \
                              --forecast_horizon $forecast_horizon \
                              --batch_size $batch_size \
                              --n_epochs $n_epochs \
                              --past_history $past_history \
                              --selected_model $selected_model \
                              --lr $lr \
                              --shuffle \
                              --kernel_size $kernel_size \
                              --num_stacks $num_stacks \
                              --num_blocks $num_blocks \
                              --fc_hidden_layers $fc_hidden_layers \
                              --fc_hidden_units $fc_hidden_units \
                              --dropout $dropout \
                              --hidden_layers_dropout $hidden_layers_dropout \
                              --dataset_name $dataset_name \
                              --num_series $num_series \
                              --dataset_num_series $dataset_num_series \
                              --data_file_path $data_file_path \
                              --exec_name $exec_name \
                              --save_model_PATH $save_model_PATH \
                              --label_len $label_len \
                              --embed_type $embed_type \
                              --embed $embed \
                              --freq $freq \
                              --activation $activation \
                              --factor $factor \
                              --d_model $d_model \
                              --n_heads $n_heads \
                              --d_ff $d_ff \
                              --d_layers $d_layers \
                              --e_layers $e_layers \
                              --approach $approach \
                              --exec_context $exec_context \
                              --exec_iterations $exec_iterations \
                              --custom_val_split $custom_val_split \
                              --invs_dataset_aug $invs_dataset_aug

      done
    done
  done





# end of the USER commands
