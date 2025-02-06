# for displaying figures in code editor
# %matplotlib inline
import matplotlib.pyplot as plt

from time import perf_counter
import numpy as np
import torch
import torch.nn as nn
import random
from Data_Store import Data_Store
import Utils.metrics as metrics
import Utils.utils as utils
import Utils.plots as plots
import Utils.tools as tools

from Models.models import get_model
from Utils.Parser import args
from Utils.clusters import getClusters
print("Pytorch version ", torch.__version__)




if torch.cuda.is_available():
    print("gpu available")



for exec_iter in range(args.exec_iterations):

    print("Current Iteration: ", exec_iter)
    # torch.manual_seed(0)
    # np.random.seed(0)
    # random.seed(0)


    cl_exuction_time  = 0
    cl_ground_truth = [[] for _ in range(args.dataset_num_series)]
    cl_y_test_preds = [[] for _ in range(args.dataset_num_series)]

    norm_cl_ground_truth = [[] for _ in range(args.dataset_num_series)]
    norm_cl_y_test_preds = [[] for _ in range(args.dataset_num_series)]
    args.current_exec_iter = exec_iter


    clusters = getClusters(args.dataset_name,args.num_series,args.approach)
    for cluster in clusters:
        args.curr_cluster = cluster
        args.num_series = len(args.curr_cluster)

        execution_time = 0

        # prepare data
        data_store = Data_Store(args)
        print("Current cluster: ", args.curr_cluster)

        # get dataloaders for training, validation and testing
        train_dataloader, val_dataloader, test_dataloader = data_store.create_datasets()

        mse_loss, mae_loss = nn.MSELoss(), nn.L1Loss()
        lst_losses = [mse_loss, mae_loss]

        # get forecasting model
        model = get_model(args).to(args.device)

        # train model
        execution_start_time = perf_counter()
        train_epochs_mae_losses, val_epochs_mae_losses, optimizer = tools.train_model(args, model, lst_losses, train_dataloader,
                                                                                      val_dataloader)
        execution_end_time = perf_counter()
        execution_time = execution_end_time - execution_start_time

        print("execution_time", execution_time)

        # test model
        pred_array, y_array = tools.test_model(args, model, lst_losses[1], test_dataloader, optimizer)

        # display training & valid losses
        plots.dispaly_losses(plt, train_epochs_mae_losses, val_epochs_mae_losses)

        # calculate normalized results
        norm_err_dic = utils.calculate_normalized_results(y_array, pred_array)

        # Denormalize results # return [num_series,batch_dim,time_dim]
        ground_truth, y_test_preds = data_store.denormalize_results(y_array, pred_array)
        #ground_truth, y_test_preds = data_store.denormalize_results_mutually(y_array, pred_array)

        #reshape normalized results to [num_series,batch_dim,time_dim]
        norm_y_array =  y_array.reshape(y_array.shape[2], y_array.shape[0], y_array.shape[1]).numpy()
        norm_pred_array =  pred_array.reshape(pred_array.shape[2],pred_array.shape[0],pred_array.shape[1]).numpy()

        cl_exuction_time = cl_exuction_time + execution_time
        for iter_index,series_idx in enumerate(args.curr_cluster):
            cl_ground_truth[series_idx] = ground_truth[iter_index]
            cl_y_test_preds[series_idx] = y_test_preds[iter_index]
            norm_cl_ground_truth[series_idx] = norm_y_array[iter_index]
            norm_cl_y_test_preds[series_idx] = norm_pred_array[iter_index]

    #shape [num_series, batch_dim, time_dim]
    cl_ground_truth = np.asarray(cl_ground_truth)
    cl_y_test_preds = np.asarray(cl_y_test_preds)
    norm_cl_ground_truth = np.asarray(norm_cl_ground_truth)
    norm_cl_y_test_preds = np.asarray(norm_cl_y_test_preds)



    # calculate denormalized results

    err_dic = metrics.displayMetrics(
        cl_ground_truth.reshape(cl_ground_truth.shape[0], cl_ground_truth.shape[1] * cl_ground_truth.shape[2]),
        cl_y_test_preds.reshape(cl_y_test_preds.shape[0], cl_y_test_preds.shape[1] * cl_y_test_preds.shape[2]))


    #retrieve initial number of series (if clustering)
    args.num_series  = args.dataset_num_series
    # plot and save figures
    plots.plot_results(args, cl_ground_truth, cl_y_test_preds)

    # save results
    utils.save_results(args=args, ground_truth=cl_ground_truth, y_test_preds=cl_y_test_preds, execution_time=cl_exuction_time,
                       err_dic=err_dic, norm_err_dic=str(norm_err_dic),norm_ground_truth=norm_cl_ground_truth,norm_y_test_preds=norm_cl_y_test_preds)
