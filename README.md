# Neural Architecture Search for Time-Series Forecasting

## Overview
This project implements a Random Search–based Neural Architecture Search (NAS)
framework to optimize LSTM and GRU models for time-series forecasting using
the Airline Passengers dataset.

## Dataset
- Source: statsmodels Airline Passengers dataset
- Characteristics: trend, seasonality, noise
- Preprocessing: Min-Max normalization and sliding window sequences

## Project Tasks Completed
- Baseline LSTM implementation
- Flexible LSTM/GRU architecture
- Random Search NAS engine
- Train / validation / test split
- Final retraining on full training set
- Evaluation using RMSE and MAE

## Setup
Install dependencies:

pip install torch numpy pandas scikit-learn statsmodels

## Run
Execute the full experiment:

python project.py

## Output
- Baseline RMSE and MAE on test set
- NAS-optimized RMSE and MAE on test set
- Final winning architecture configuration

## Notes
The test set is never used during NAS to ensure unbiased evaluation.
