# TS_Forecasting_Project

# Time Series Forecasting with Multiple Neural Network Architectures

This repository provides an in-depth comparison of both existing and novel neural network architectures on the task of time series forecasting. The models are evaluated across multiple datasets, and the performance metrics are recorded for each model and dataset combination.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Models Compared](#models-compared)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Time series forecasting is a critical problem in many domains, such as finance, weather prediction, and energy consumption. This project aims to compare the performance of various neural network architectures, including traditional models like LSTM and GRU, as well as novel architectures designed for this task.

The following neural networks are evaluated on several publicly available time series datasets:

- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Units)**
- **Transformer-based architectures**
- **Temporal Convolutional Networks (TCN)**
- **Attention-based models**
- **Other novel architectures**

The goal is to determine which architecture performs best across different time series data characteristics, such as seasonality, trend, and noise levels.

## Project Structure

The directory structure of the project is as follows:

├── Checkpoint_file ├── Data_Store.py ├── Datasets │ └── (Contains different time series datasets used for training and testing) ├── Models │ └── (Different neural network model architectures implemented here) ├── Results │ └── (Output results and performance metrics stored here) ├── clustering_files ├── Pytorch_Main.py ├── exec_1 ├── TF ├── Utils │ └── (Helper functions and utility scripts) ├── run.ipynb ├── run_all.sh ├── Figures(Travaux) │ └── (Plots and visualizations for the results) ├── LCA Figures.ipynb ├── Plot_Results.ipynb └── README.md


## Installation

To run the code, you will need Python 3.x and the required dependencies specified in the `requirements.txt` file.

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/timeseries-forecasting-comparison.git
cd timeseries-forecasting-comparison

pip install -r requirements.txt



### Instructions for GitHub:
1. Create or open the `README.md` file in your repository.
2. Paste the markdown content provided above.
3. Commit the changes.

This README is now in proper GitHub markdown format, ready to be displayed with proper headers, code blocks, and bullet points. Let me know if you need further adjustments!

 
