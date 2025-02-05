# TS_Forecasting_Project

# Time Series Forecasting with Multiple Neural Network Architectures

This repository provides a comprehensive comparison of both existing and novel neural network architectures for time series forecasting, implemented across three forecasting approaches: local, global, and clustering-based forecasting. The models are evaluated on multiple datasets, with performance metrics recorded for each model, dataset, and forecasting approach combination.

## Overview

Time series forecasting is a critical problem in many domains, such as finance, weather prediction, and energy consumption. This project aims to compare the performance of various neural network architectures, including traditional models like LSTM and GRU, as well as novel architectures designed for this task.

The following neural networks are evaluated on several publicly available time series datasets:

- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Units)**
- **Transformer-based architectures**
- **LTSF-Linear (NLinear,DLinear,Linear) (TCN)**
- **Attention-based models**
- **CNN_N_BEATS**
- **BEATS_CELL**
- **RLinear**
- **RMLP**
- **N-BEATS**
- **Multivariate N-BEATS**
- **SegRNN**
- **Other novel architectures**

The goal is to determine which architecture performs best across different time series data characteristics, across three forecasting approaches (local, global and clustering). For the forecasting approach, the clustrering algorthms used include K-Means, Optics and SOM algorithms (found in /clustering_files). 

## Datasets 

This project evaluates the models on multiple time series datasets. These datasets are stored in the Datasets/ folder. You can download or place your time series datasets in this folder, ensuring they follow the appropriate format.

Datasets used include:

Energy Consumption Data
- electricity
- ETTh1.csv
- ETTh2.csv
- ETTm1.csv
- ETTm2.csv
  
Stock Market Data
- exchange_rate.csv
  
Traffic Data
- traffic.csv
  
Weather Data
- weather.csv


## Installation

To run the code, you will need Python 3.x and the required dependencies specified in the `requirements.txt` file.

To run code on CPU (default) : 
- bash run_all.sh

To run code on GPU : 
- specify GPU support in exec_1/main_exec.sh
- bash run_all 

 #Citing 
 
 @inproceedings{Zeng2022AreTE,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
