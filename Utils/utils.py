
import json
import numpy as np
import Utils.metrics as metrics
import torch
from pathlib import Path


def retrieve_best_training_models(model, checkpoint_dir):
    model.load_weights(checkpoint_dir)
    # results = model.evaluate(test_dataset)
    return model

def save_weights(model, checkpoint_dir):
    model.save_weights(checkpoint_dir)

def restore_saved_weights(model, checkpoint_dir):
    model.load_weights(checkpoint_dir)
    # results = n_beats_model.evaluate(test_dataset)
    return model

def save_entire_model(model, checkpoint_dir='Checkpoint_file/models/my_model.h5'):
    import keras
    # save model
    model.save(checkpoint_dir)
    # load model
    keras.models.load_model(checkpoint_dir)
    model.summary()


# flatten lists
def save_results_Incsv(csv, file_name, flattened_results):
    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(flattened_results)

# save matrices

# save execution parameters

def save_dic(args,exec_name,execution_time, err_dic, norm_err_dic):

    exec_dict = {
        "exec_name": args.exec_name,
        "training_size": args.train_size,
        "validation_size": args.val_size,
        "testing_size": args.test_size,
        "forecast_horizon": args.forecast_horizon,
        "past_history": args.past_history,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "n_epochs": args.n_epochs,
        "num_series": args.num_series,
        "kernel_size": args.kernel_size,
        "num_stacks": args.num_stacks,
        "num_blocks": args.num_blocks,
        "fc_hidden_layers": args.fc_hidden_layers,
        "fc_hidden_units": args.fc_hidden_units,
        "block_sharing": args.block_sharing,
        "dropout": args.dropout,
        "execution_time":execution_time,
        "err_results": err_dic,
        "norm_err_dic": norm_err_dic,
        "forecasting_approach":args.approach
    }
    # print(exec_dict)
    with open(exec_name + ".txt", 'w') as convert_file:
        convert_file.write(json.dumps(exec_dict))
        print('dictionary saved successfully to file')

# save results(labels & preds)
def save_matrices(gt, preds, fn_labels, fn_preds):
    np.savetxt(fn_labels + ".csv", gt, delimiter=",")
    np.savetxt(fn_preds + ".csv", preds, delimiter=",")

def create_dir(path):
    try:
        path = Path(path)
        path.mkdir(parents=True,exist_ok=True)
    except OSError as error:
         print(error)


#save model
def saveModel(model, N_EPOCHS, optimizer, PATH, loss=0.4):
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)

#load saved model
def loadModel(model, PATH, optimizer):
    #print("loading model: ", PATH)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Loaded model : epoch {} - Loss {}: ".format(epoch,loss))
    #model.eval()
    # - or -
    # model.train()

def calculate_normalized_results(y_array,pred_array):
    #normalized errors
    norm_err_dic= {
            "mse": metrics.mse(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "mae": metrics.mae(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "rmse": metrics.rmse(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "wape": metrics.wape(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "mape": metrics.mape(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) ),
            "r2": metrics.r2_avr(np.transpose(y_array.numpy(), (2, 0, 1) ),np.transpose(pred_array.numpy(), (2, 0, 1)) )
        }
    return norm_err_dic

#save results and execution parameters
def save_results(args,ground_truth,y_test_preds, execution_time,err_dic,norm_err_dic,norm_ground_truth,norm_y_test_preds):
    #flatten matrices, colapse batch dim
    gt_save  = ground_truth.reshape(ground_truth.shape[0],ground_truth.shape[1]*ground_truth.shape[2])
    preds_save  = y_test_preds.reshape(y_test_preds.shape[0],y_test_preds.shape[1]*y_test_preds.shape[2])

    norm_gt_save  = norm_ground_truth.reshape(norm_ground_truth.shape[0],norm_ground_truth.shape[1]*norm_ground_truth.shape[2])
    norm_preds_save  = norm_y_test_preds.reshape(norm_y_test_preds.shape[0],norm_y_test_preds.shape[1]*norm_y_test_preds.shape[2])

    #create results' folder for dataset
    location = "Results/"+args.dataset_name+"/"+args.selected_model+"/"+args.exec_context+'/denorm_results'
    location_norm = "Results/"+args.dataset_name+"/"+args.selected_model+"/"+args.exec_context+'/norm_results'
    create_dir(location)
    create_dir(location_norm)

    # #save execution hyperparameters
    save_dic(args,exec_name=location+"/"+"iter"+str(args.current_exec_iter)+"_"+args.approach+'_'+args.exec_name,execution_time=execution_time,err_dic=err_dic,norm_err_dic=str(norm_err_dic))

    #save results(labels & preds)
    save_matrices(gt_save,preds_save,location+"/"+"iter"+str(args.current_exec_iter)+"_"+args.approach+"_labels_"+args.exec_name,location+"/"+"iter"+str(args.current_exec_iter)+"_"+args.approach+"_preds_"+args.exec_name)

    save_matrices(norm_gt_save,norm_preds_save,location_norm+"/"+"iter"+str(args.current_exec_iter)+"_"+args.approach+"_labels_"+args.exec_name,location_norm+"/"+"iter"+str(args.current_exec_iter)+"_"+args.approach+"_preds_"+args.exec_name)


def saveArrayInExcelFormat(num_array,pd,file_path='file.xlsx'):
    df = pd.DataFrame (num_array)
    df.to_excel(file_path, index=False)


class standard_scaler():
    def __init__(self, ts, sub_last=False, cat_std=False):
        self.sub_last = sub_last
        self.cat_std = cat_std
        self.mean = ts.mean(-1, keepdim=True)
        self.std = torch.sqrt(torch.var(ts-self.mean, dim=-1, keepdim=True, unbiased=False) + 1e-5)

    def transform(self, data):
        if self.sub_last:
            self.last_value = data[...,-1:].detach()
            data = data - self.last_value
        data = (data - self.mean) / self.std
        if self.cat_std:
            data = torch.cat((data, self.mean, self.std),-1)
        return data

    def inverted(self, data):
        if self.cat_std:
            data =  data[...,:-2] * data[...,-1:] + data[...,-2:-1]
        else:
            data = (data * self.std) + self.mean
        data = data + self.last_value if self.sub_last else data
        return data