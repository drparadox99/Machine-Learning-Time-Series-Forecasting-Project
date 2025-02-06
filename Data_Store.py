#import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class Data_Store():
    def __init__(self, args):
        self.args = args
        self.data_file_path = args.data_file_path
        self.df = []
        self.dataset = []
        self.training_percentage = args.train_size
        self.testing_percentage = args.test_size
        self.validation_percentage = args.val_size
        self.past_history = args.past_history
        self.forecast_horizon = args.forecast_horizon
        self.batch_size = args.batch_size
        self.test_data = []
        self.scalers = []
        self.mutual_scaler = ""
        self.shuffle = args.shuffle

    # import dataset
    def importData(self, file_path):
        if self.data_file_path.endswith('csv'):
            series_df = pd.read_csv(file_path)
        else:
            series_df = pd.read_excel(file_path)
        return series_df #(samples,num_series)

    # drop useless columns and rows
    def exstract_dataset(self, data,col_name="date"):
        # dataset: (samples,series)
        dataset = data.drop(col_name, axis=1)
        dataset = dataset.to_numpy()
        return dataset #dataset: (samples,series)

        # split into train and test dataset

    def split_dataset(self, dataset, index_split):
        # dataset: (samples,series)
        train_dataset = dataset[:-index_split]
        test_dataset = dataset[-index_split:]
        # return: (samples,series)
        return train_dataset, test_dataset

        # normalize and denormalize multivariate dataset collectively

    def normalize_mutually(self, normalization, scaler_object, training_dataset=[], testing_dataset=[]):
        # data (samples,num_series)

        transformed_training_set = 0
        transformed_testing_set = 0
        if normalization is True:
            scaler_object.fit(training_dataset)
            transformed_training_set = scaler_object.transform(training_dataset)
            transformed_testing_set = scaler_object.transform(testing_dataset)
        else:
            transformed_testing_set = scaler_object.inverse_transform(testing_dataset)
        return transformed_training_set, transformed_testing_set

    # fit & normalize multivariate training dataset seperately
    def normalizeTrainDataset(self, dataset):
        # dataset: (series,samples)
        norm_seqs = []
        lst_scalers = []

        for series in dataset:
            scaler = preprocessing.MinMaxScaler()
            #scaler = preprocessing.StandardScaler()
            # reshape input into numpy 2 D
            reshaped_2d_series = series.reshape(-1, 1)
            scaler.fit(reshaped_2d_series)
            norm_series = scaler.transform(reshaped_2d_series)
            # norm_pseudoSequence = norm_pseudoSequence.squeeze().float()    #reshape back to 1 D
            norm_series = norm_series.reshape(-1, )  # reshape back to 1 D

            norm_seqs.append(norm_series)
            lst_scalers.append(scaler)
        # return: (samples,series)
        return np.asarray(norm_seqs).T, lst_scalers

    # normalize and denormalize multivariate testing dataset seperately
    def normalizeTestDataset(self, dataset, lst_scalers, normalization):
        # dataset: (series,samples)
        norm_seqs = []
        for index, series in enumerate(dataset):
            scaler = lst_scalers[index]
            if normalization is True:
                # reshape input into numpy 2 D
                reshaped_2d_series = series.reshape(series.shape[0], 1)
                norm_series = scaler.transform(reshaped_2d_series)
                norm_series = norm_series.reshape(norm_series.shape[0], )  # reshape back to 1 D
                norm_seqs.append(norm_series)
            else:  # denormalization
                # temp = series.reshape(-1,1) #if 1D
                temp = series
                temp = scaler.inverse_transform(temp)
                # temp = temp.reshape(-1,)
                norm_seqs.append(temp)
        # return: (samples,series)
        return np.asarray(norm_seqs).T

    # denormalize multivariate testing dataset seperately
    def denormalize_results(self, y_array, pred_array):
        ground_truth_denorm = self.normalizeTestDataset(y_array.permute(2, 0, 1), self.scalers, False)
        preds_denorm = self.normalizeTestDataset(pred_array.permute(2, 0, 1), self.scalers, False)
        ground_truth = np.transpose(ground_truth_denorm, (2, 1, 0))  # shape to (num_series,batch_dim,time_dim)
        y_test_preds = np.transpose(preds_denorm, (2, 1, 0))  # shape to (num_series,batch_dim,time_dim)
        return ground_truth,y_test_preds

    def denormalize_results_mutually(self,y_array, pred_array ):
        B = y_array.shape[0]
        time_dim = y_array.shape[1]
        _, ground_truth = self.normalize_mutually(False, scaler_object=self.mutual_scaler,
                                                        testing_dataset=y_array.reshape(y_array.shape[0]*y_array.shape[1],y_array.shape[2]))
        _, y_test_preds = self.normalize_mutually(False, scaler_object=self.mutual_scaler,
                                                        testing_dataset=pred_array.reshape(pred_array.shape[0] * pred_array.shape[1], pred_array.shape[2]))
        ground_truth = np.transpose(ground_truth.reshape(B,time_dim,ground_truth.shape[1]),(2,0,1))
        y_test_preds = np.transpose(y_test_preds.reshape(B,time_dim,y_test_preds.shape[1]),(2,0,1))
        return ground_truth,y_test_preds


    # split based on frequency, without y duplicates #lookback window is day_to_predict * forecast_horizon # univariate series & last ragged values not discarded
    def build_testing_data(self, test_data, forecast_horizon, day_to_predict, days_history):
        # group list by forecast horizon(days)
        list_grouped_by_days = [test_data[n:n + forecast_horizon] for n in range(0, len(test_data), forecast_horizon)]
        start_day_index = 0
        end_day_index = days_history
        x_test = []
        y_test = []
        num_preds = len(list_grouped_by_days) - days_history  # estimate number of predictions
        # predict num_preds by splitting to x_test and y_test
        for jour in range(num_preds):
            start_day_index = day_to_predict - days_history  # start day index
            end_day_index = day_to_predict  # end day index

            # flatten multiple lists
            flattened_list = list_grouped_by_days[start_day_index: end_day_index]
            flattened_list = [item for sublist in flattened_list for item in sublist]
            # flattened_list = flattened_list[:-8] #Enlever les 4 dernières heures (4 * 2)
            x_test.append(flattened_list)
            y_test.append(list_grouped_by_days[end_day_index])
            day_to_predict = day_to_predict + 1  # increemnt day to predict
        return x_test, y_test

    # for both multivariate & univariate split with y duplicates
    def build_mutlivariate_training_data(self, x_dataset, past_history, forecast_horizon):
        # dataset: (num_samples,num_series)
        x = []
        y = []
        num_samples = len(x_dataset)
        num_possible_split_iter = ((num_samples - past_history) + 1) - forecast_horizon
        # start_index_forecast  = past_history   #first start index of y labels
        # end_index_forecast = (start_index_forecast + forecast_horizon) - 1
        for i in range(0, num_possible_split_iter):
            x.append(x_dataset[i:i + past_history])
            y.append(x_dataset[i + past_history: (i + past_history) + forecast_horizon])
        return np.asarray(x), np.asarray(y)

    # for both multivariate & univariate split without y duplicates
    def split_multivariate_dataset_wihtout_duplicates(self, x_dataset, past_history, forecast_horizon):
        # dataset: (num_samples,num_series)
        import math
        x = []
        y = []
        num_samples = len(x_dataset)  # x_dataset(num_samples,num_series)
        num_possible_split_iter = math.floor((num_samples - past_history) / forecast_horizon)
        start_index_forecast = past_history  # first start index of y labels
        end_index_forecast = (start_index_forecast + forecast_horizon) - 1  # first end index of y labels
        num_last_vals_discarded = (num_samples - past_history) % forecast_horizon + 1
        print("num_last_val_discarded : ", num_last_vals_discarded)
        for i in range(0, num_possible_split_iter):
            x.append(x_dataset[start_index_forecast - past_history:start_index_forecast])  # x values
            y.append(x_dataset[start_index_forecast:end_index_forecast + 1])  # y values
            start_index_forecast = end_index_forecast + 1
            end_index_forecast = (start_index_forecast + forecast_horizon) - 1
        return np.asarray(x), np.asarray(y)

    # pytorch
    class Custom_Dataset(Dataset):
        def __init__(self, X, Y):
            """
            Initialize the custom dataset.

            Args:
                X (list or numpy.ndarray): Input data.
                Y (list or numpy.ndarray): Target data.
            """
            self.X = torch.tensor(X, dtype=torch.float32)  # Convert X to PyTorch tensor
            self.Y = torch.tensor(Y, dtype=torch.float32)  # Convert Y to PyTorch tensor

        def __len__(self):
            """
            Return the number of samples in the dataset.
            """
            return len(self.X)

        def __getitem__(self, idx):
            """
            Get a sample from the dataset at the given index.

            Args:
                idx (int): Index of the sample.

            Returns:
                x (tensor): Input data for the sample.
                y (tensor): Target data for the sample.
            """
            x = self.X[idx]
            y = self.Y[idx]
            return x, y


    def ts_flip(self,dataset):
        #dataset: [samples,num_series]
        flipped_lst = []
        for series in dataset.T:
            flipped_ts = np.flip(series)
            s_concat = np.concatenate((series, flipped_ts), axis=-1)
            flipped_lst.append(s_concat)
        dataset = np.asarray(flipped_lst).T
        return dataset

    def create_datasets(self):
        dataset = self.importData(self.data_file_path)

        # dataset = self.exstract_dataset(dataset).astype(float)
        # for series in range(len( dataset.T)):
        #     dataset.T.iloc[series,:].plot()

        dataset = self.exstract_dataset(dataset) # convert to numpy (samples,num_series)

        #retrieve only time-series present in cluster
        if self.args.approach == "local" or self.args.approach == "clustering":
            cl_dataset = np.asarray([dataset.T[series_idx] for series_idx in self.args.curr_cluster]).T
            dataset = cl_dataset

        # dataset = np.stack((np.arange(7588), np.arange(7588)+2,np.arange(7588)+4),axis=1) ###
        # dataset = dataset.T[0:1,:].T

        training_split = int(len(dataset) * self.training_percentage)
        testing_split = int(len(dataset) * self.testing_percentage)
        validation_split = int(len(dataset) * self.validation_percentage)
        final_testing_split = testing_split + self.past_history
        training_dataset, testing_dataset = dataset[:-final_testing_split], dataset[-final_testing_split:]  # (5883,8) (test_x,test_y)


        if (self.args.invs_dataset_aug == 1):
            print("adding inverse dataset aug ...")
            training_dataset = self.ts_flip(training_dataset)

        # normalization
        training_dataset, self.scalers = self.normalizeTrainDataset(training_dataset.T)
        testing_dataset = self.normalizeTestDataset(testing_dataset.T, self.scalers, True)  # à voir

        #self.mutual_scaler =  preprocessing.MinMaxScaler()
        #training_dataset, testing_dataset = self.normalize_mutually(True,self.mutual_scaler,training_dataset,testing_dataset)
        #_, testing_dataset = self.normalize_mutually(False,self.mutual_scaler,training_dataset,testing_dataset)

        # build_mutlivariate_training_data
        if self.args.training_duplicates == 0:
            X_train, y_train = self.split_multivariate_dataset_wihtout_duplicates(x_dataset=training_dataset,past_history=self.past_history,forecast_horizon=self.forecast_horizon)
        else:
            X_train, y_train = self.build_mutlivariate_training_data(x_dataset=training_dataset,past_history=self.past_history,forecast_horizon=self.forecast_horizon)


        if (self.args.custom_val_split == 1):
            print("custon validation split...")
            X_train, x_val, y_train, y_val = self.sequential_val_split(X_train, y_train, self.validation_percentage)
        else:
            X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_percentage, random_state=42,shuffle=False)
            print("classic validation split")

        if self.args.testing_duplicates == 0:
            x_test, y_test = self.split_multivariate_dataset_wihtout_duplicates(x_dataset=testing_dataset,past_history=self.past_history,forecast_horizon=self.forecast_horizon)
        else:
            x_test, y_test = self.build_mutlivariate_training_data(x_dataset=testing_dataset,past_history=self.past_history,forecast_horizon=self.forecast_horizon)

        # denormalizing labels
        # y_train = self.normalizeTestDataset(y_train.T,self.scalers,False)
        # y_test = self.normalizeTestDataset(y_test.T,self.scalers,False)
        # y_val = self.normalizeTestDataset(y_val.T,self.scalers,False)

        print(np.asarray(X_train).shape)
        print(np.asarray(y_train).shape)
        print(np.asarray(x_val).shape)
        print(np.asarray(y_val).shape)
        print(np.asarray(x_test).shape)
        print(np.asarray(y_test).shape)


        # cumstom datasets
        train_dataset = self.Custom_Dataset(X_train, y_train)
        validation_dataset = self.Custom_Dataset(x_val, y_val)
        test_dataset = self.Custom_Dataset(x_test, y_test)

        # create dataloaders
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=self.shuffle, drop_last=False)
        validation_dataloader = DataLoader(validation_dataset, self.batch_size, drop_last=False)
        test_dataloader = DataLoader(test_dataset, self.batch_size, drop_last=False)

        return train_dataloader, validation_dataloader, test_dataloader  # , y_test_evaluation


    # Generate multiple train-validation splits in a sequential manner, each validation set comes after its corresponding training set
    def sequential_val_split(self, X, Y, val_size):
        # X & Y[Batch,time_dim,channels]
        len_ = len(X)
        val_s = int(val_size * len_)
        # generate n random indexes between ranges
        rand_idx = random.sample(range(0, len_), val_s)
        x_val, y_val, x_train, y_train = [], [], [], []
        for batch_idx, series in enumerate(X):
            if batch_idx in rand_idx:  # add to validation set
                x_val.append(X[batch_idx])
                y_val.append(Y[batch_idx])
            else:  # add to training set
                x_train.append(X[batch_idx])
                y_train.append(Y[batch_idx])

        x_val, y_val = np.asarray(x_val), np.asarray(y_val)
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        return x_train, x_val, y_train, y_val

