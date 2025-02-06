import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PyT.Final_Models import DC_BEATS,Deep_Blocks
from PyT.LTSF_Linear import DLinearModel,Simple_Linear,N_Linear

def show(_sting, content):
    print(_sting + str(content))



class DC_DLinear_M():
    def __init__(
            self,
            input_size: tuple,
            output_size: int,
            kernel_size: int,
            num_sub_models: int,
            individual: bool,
            training_dataloader: DataLoader,
            testing_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            num_epochs,
            lst_losses,
            learning_rate,
            PATH,
            save_optimal_model,
            model_autoformer,
            device,
            metrics,
            num_stacks,
            num_blocks,
            fc_hidden_layers,
            fc_hidden_units,
            block_sharing,
            enable_theta_basis,
            thetas_dim,
            dropout
    ):
        self.sub_forecasting_horizon = int(output_size / num_sub_models)
        self.num_sub_models = num_sub_models
        self.device = device
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.lst_losses = lst_losses
        self.PATH = PATH
        self.save_optimal_model = save_optimal_model

        #autoformer
        self.model_autoformer = model_autoformer

        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.validation_dataloader = validation_dataloader

        #dlinear
        self.individual = individual

        #DLinearBeats
        self.fc_hidden_layers = fc_hidden_layers
        self.fc_hidden_units = fc_hidden_units

        self.sub_models = nn.ModuleList([
            # DLinearModel(
            #     kernel_size=kernel_size,
            #     seq_len=input_size[0],
            #     pred_len=self.sub_forecasting_horizon,
            #     individual=self.individual,
            #     num_series=input_size[1]
            # )
            # DLinearBeats_M(
            #     input_size=input_size,
            #     output_size=self.sub_forecasting_horizon,
            #     kernel_size=kernel_size,
            #     num_stacks=num_stacks,
            #     num_blocks=num_blocks,  # essayer 4
            #     fc_hidden_layers = self.fc_hidden_layers,
            #     fc_hidden_units = self.fc_hidden_units,
            #     block_sharing=block_sharing,
            #     enable_theta_basis= enable_theta_basis,
            #     thetas_dim= thetas_dim,
            #     dropout=dropout
            # )
            # Simple_Linear(
            #     input_size=input_size,
            #     forecasting_horizon=self.sub_forecasting_horizon
            # )
            # N_Linear(input_size=input_size,
            #          forecast_horizon=self.sub_forecasting_horizon,
            #          individual=False)
            # Deep_Blocks(
            #     input_size=input_size,
            #     forecast_horizon=self.sub_forecasting_horizon,
            #     num_blocks=num_blocks,
            #     num_fc_layers= fc_hidden_layers,
            #     expansion_dim= fc_hidden_units,
            #     dropout= dropout
            # )
            DC_BEATS(
                    input_size=input_size,
                    output_size=self.sub_forecasting_horizon,
                    kernel_size=kernel_size,
                    num_stacks=num_stacks,
                    num_blocks=num_blocks,  # essayer 4
                    fc_hidden_layers = self.fc_hidden_layers,
                    fc_hidden_units = self.fc_hidden_units,
                    block_sharing=block_sharing,
                    dropout=dropout
            )

            for i in range(self.num_sub_models)
        ])
        self.sub_models_optim = []
        self.sub_models_PATH = [[] for i in range(self.num_sub_models)]
        self.sub_models_losses = [[[],[]] for i in range(self.num_sub_models)]



    def train_sub_models(self):

        sm_y_start = 0
        sm_y_end = self.sub_forecasting_horizon
        for model_index, s_model in enumerate(self.sub_models):
            #optimizer = torch.optim.Adam(s_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08,weight_decay=0)
            optimizer = torch.optim.Adam(s_model.parameters(), lr=self.learning_rate)
            self.sub_models_optim.append(optimizer) #save optimizer for each sub_model
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.0001, patience=50, verbose=True)
            print("---------------------- Training sub model n° {} ---------------------- ".format(model_index))
            self.sub_models_losses[model_index][0],self.sub_models_losses[model_index][1] = self.train_model(model_index,s_model, self.num_epochs,
                             self.lst_losses, optimizer,lr_scheduler, self.model_autoformer, self.device,self.metrics, sm_y_start, sm_y_end)
            sm_y_start += self.sub_forecasting_horizon
            sm_y_end += self.sub_forecasting_horizon


    def test_sub_models(self):
        flattened_preds = []
        flattened_ys = []
        sm_y_start = 0
        sm_y_end = self.sub_forecasting_horizon
        for model_index in range(self.num_sub_models):
            if self.save_optimal_model is True:
                self.loadModel(self.sub_models[model_index], self.sub_models_PATH[model_index], self.sub_models_optim[model_index]) #load sub_model
            print("---------------------- Testing sub model n° {} ---------------------- ".format(model_index))
            flattened_pred, flattened_y = self.test_model(model_index,self.sub_models[model_index], self.lst_losses[1],
                                                          self.device, self.metrics, sm_y_start, sm_y_end)
            sm_y_start += self.sub_forecasting_horizon
            sm_y_end += self.sub_forecasting_horizon
            flattened_preds.append(flattened_pred)
            flattened_ys.append(flattened_y)

        concat_flattened_preds = torch.cat(flattened_preds, axis=1)
        concat_flattened_ys = torch.cat(flattened_ys, axis=1)

        return concat_flattened_preds, concat_flattened_ys

    #
    # def train_model(self,model_index, model, num_epochs, lst_losses, optimizer,lr_scheduler, model_autoformer, device, metrics, sm_y_start, sm_y_end):
    #     best_metric = float('inf')  # Initialize with positive infinity for loss
    #     train_epochs_mae_losses = []
    #     train_epochs_mse_losses = []
    #     val_epochs_mae_losses = []
    #     val_epochs_mse_losses = []
    #     label_len = 48
    #
    #     for epoch in range(num_epochs):
    #         model.train()
    #         epoch_mse_loss = 0
    #         epoch_mae_loss = 0
    #         for batch, item in enumerate(self.training_dataloader):
    #
    #             x, y = item
    #             y = y[:, sm_y_start:sm_y_end, :]
    #             n_features = x.shape[2] if x.ndim == 3 else 1
    #             x = x.reshape(x.shape[0], x.shape[1], n_features)  # reshape to 3D
    #             x = x.to(device)  # for GPU
    #             y = y.to(device)  # for GPU
    #             if model_autoformer:
    #                 dec_inp = y[:, -label_len:, :]  # get last (48,num_series) lines of each batch of the input data
    #                 dec_inp_zeros = torch.zeros_like(
    #                     y[:, -y.shape[1]:, :]).float()  # mask of zeros of forecasting shape
    #                 dec_inp = torch.cat([dec_inp, dec_inp_zeros], dim=1).float()
    #                 pred = model(x_enc=x, x_dec=dec_inp)
    #                 print("after training my model...")
    #             else:
    #                 pred = model(x)  # generate preds
    #             optimizer.zero_grad()  # zero out gradients
    #
    #             mse_loss = lst_losses[0](pred, y)  # compute mse loss
    #             mae_loss = metrics.mae(pred.detach().numpy(), y.numpy())  # compute mae loss
    #
    #             mse_loss.backward()  # calculate mse gradients by backprop
    #             optimizer.step()  # update mse gradients
    #
    #             epoch_mse_loss += mse_loss.item()  # sum up losses
    #             epoch_mae_loss += mae_loss  # sum up losses
    #
    #         avg_mse_training_loss = epoch_mse_loss / len(self.training_dataloader)
    #         avg_mae_training_loss = epoch_mae_loss / len(self.training_dataloader)
    #         train_epochs_mse_losses.append(avg_mse_training_loss)  #
    #         train_epochs_mae_losses.append(avg_mae_training_loss)  #
    #
    #         # Step 6: Validation loop
    #         model.eval()  # Set the model in evaluation mode
    #         val_epoch_mse_loss = 0
    #         val_epoch_mae_loss = 0
    #
    #         with torch.no_grad():
    #             for batch, item in enumerate(self.validation_dataloader):
    #                 x, y = item
    #                 n_features = x.shape[2] if x.ndim == 3 else 1
    #                 x = x.reshape(x.shape[0], x.shape[1], n_features)  # reshape to 3D
    #                 x = x.to(device)  # for GPU
    #                 y = y.to(device)  # for GPU
    #                 y = y[:, sm_y_start:sm_y_end, :]
    #
    #                 if model_autoformer:
    #                     dec_inp = y[:, -label_len:, :]  # get last (48,num_series) lines of each batch of the input data
    #                     dec_inp_zeros = torch.zeros_like(
    #                         y[:, -y.shape[1]:, :]).float()  # mask of zeros of forecasting shape
    #                     dec_inp = torch.cat([dec_inp, dec_inp_zeros], dim=1).float()
    #                     output = model(x_enc=x, x_dec=dec_inp)
    #                 else:
    #                     output = model(x)
    #
    #                 mse_loss = lst_losses[0](output, y)  # mse loss
    #                 mae_loss = lst_losses[1](output, y)  # mae loss
    #                 val_epoch_mse_loss += mse_loss.item()
    #                 val_epoch_mae_loss += metrics.mae(output.detach().numpy(), y.numpy())  # 4 GPU
    #
    #         avg_mse_val_loss = val_epoch_mse_loss / len(self.validation_dataloader)
    #         avg_mae_val_loss = val_epoch_mae_loss / len(self.validation_dataloader)
    #
    #         # lr_scheduler.step(avg_mae_val_loss)
    #         val_epochs_mse_losses.append(avg_mse_val_loss)
    #         val_epochs_mae_losses.append(avg_mae_val_loss)
    #         print(
    #             f'Epoch [{epoch + 1}/{num_epochs}] Epoch {epoch + 1}: mse:- {avg_mse_training_loss:.5f} - mae:- {avg_mae_training_loss:.5f} - val_mse: {avg_mse_val_loss:.5f} - val_mae: {avg_mae_val_loss:.5f}')
    #         if self.save_optimal_model is True:
    #             if (avg_mae_val_loss < best_metric):  # save best model
    #                 print(
    #                     "Epoch [{}/{}] : val_loss improved from {:.5f} to {:.5f}, saving model to Checkpoint_file".format(
    #                         epoch + 1, num_epochs, best_metric, avg_mae_val_loss))
    #                 best_metric = avg_mae_val_loss
    #
    #                 self.sub_models_PATH[model_index] = self.PATH+"model_"+str(model_index)  #save path to sub_model
    #                 self.saveModel(model, epoch + 1, optimizer, self.sub_models_PATH[model_index]) #save sub_model object
    #
    #     return train_epochs_mae_losses, val_epochs_mae_losses
    #
    # def test_model(self,model_index, model, mae_loss, device, metrics, sm_y_start, sm_y_end):
    #     model.eval()
    #     test_loss = 0
    #     r2_score = 0
    #     flattened_pred = []
    #     flattened_y = []
    #     with torch.no_grad():
    #         for batch, item in enumerate(self.testing_dataloader):  # 8 batches of 64
    #             x, y = item
    #             y = y[:, sm_y_start:sm_y_end, :]
    #             if (batch < 1):
    #                 print("--P_h {} - F_h {}--".format(x.shape[1], y.shape[1]))
    #             n_features = x.shape[2] if x.ndim == 3 else 1
    #             x = x.reshape(x.shape[0], x.shape[1], n_features)  # reshape to 3D
    #             x = x.to(device)
    #             y = y.to(device)
    #             pred = model(x)
    #             pred = pred.detach().cpu()
    #             y = y.detach().cpu()
    #             flattened_pred.append(pred)
    #             flattened_y.append(y)
    #             loss = mae_loss(pred, y)
    #             test_loss += loss.item()
    #             r2_score += metrics.r2_avr(pred.permute(2,0,1), y.permute(2,0,1))
    #             # print(f'Testing - Batch Loss: {loss.item()}')
    #             # print(f'Testing - Batch R2 Loss: {metrics.r2_score(pred,y)}')
    #     print(
    #         f'Testing - Average Loss: {test_loss / len(self.testing_dataloader)} - Average R2 Loss: {r2_score / len(self.testing_dataloader)}')
    #     flattened_pred = torch.cat(flattened_pred, dim=0)  # concat all pred batches
    #     flattened_y = torch.cat(flattened_y, dim=0)  # concat all label batches
    #
    #     return flattened_pred, flattened_y
    #
    # def saveModel(self,model, N_EPOCHS, optimizer, PATH, loss=0.4):
    #     torch.save({
    #         'epoch': N_EPOCHS,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss,
    #     }, PATH)
    #
    # def loadModel(self,model, PATH, optimizer):
    #     #print("loading model: ", PATH)
    #     checkpoint = torch.load(PATH)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']
    #     print("Loaded model : epoch {} - Loss {}: ".format(epoch,loss))
    #     #model.eval()
    #     # - or -
    #     # model.train()

