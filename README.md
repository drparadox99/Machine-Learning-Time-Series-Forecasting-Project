# TS_Forecasting_Project

# Time Series Forecasting with Multiple Neural Network Architectures

This repository provides an in-depth comparison of both existing and novel neural network architectures on the task of time series forecasting. The models are evaluated across multiple datasets, and the performance metrics are recorded for each model and dataset combination.

## Overview

Time series forecasting is a critical problem in many domains, such as finance, weather prediction, and energy consumption. This project aims to compare the performance of various neural network architectures, including traditional models like LSTM and GRU, as well as novel architectures designed for this task.

The following neural networks are evaluated on several publicly available time series datasets:

- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Units)**
- **Transformer-based architectures**
- **LTSF-Linear (NLinear,DLinear,Linear) (TCN)**
- **Attention-based models**
- **SegRNN**
- **Other novel architectures**

The goal is to determine which architecture performs best across different time series data characteristics, such as seasonality, trend, and noise levels.

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



 
