# Predictive Maintenance System

A machine learning-based predictive maintenance system for estimating
Remaining Useful Life (RUL) of turbofan engines and identifying engines
that require maintenance.

The project combines data preprocessing, feature engineering, classical
machine learning, deep learning, prediction, and maintenance scheduling
through a Streamlit application.

## Project Overview

Unexpected equipment failure can result in costly downtime and maintenance.
Predictive maintenance aims to estimate the future health of equipment and
allow maintenance to be performed before failure occurs.

This project uses the NASA C-MAPSS turbofan engine degradation dataset to
develop models that estimate Remaining Useful Life (RUL).

The final system provides:

- Data preprocessing
- Feature engineering
- RUL prediction
- Engine health classification
- Critical engine identification
- Maintenance scheduling
- Streamlit-based user interface
- Microsoft Outlook calendar integration

## System Pipeline

```text
Raw Sensor Data
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Machine Learning / Deep Learning Models
       ↓
RUL Prediction
       ↓
Health Status
       ↓
Maintenance Decision
       ↓
Outlook Scheduling


Models

Several models were investigated during development:

Random Forest
XGBoost
GRU
stacked GRU architecture
Stacked LSTM architecture

The notebooks document the development and evaluation process for each
model.

Dataset

The project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion
System Simulation) turbofan engine degradation dataset FD001.

The dataset contains:

Engine/unit identifiers
Operating cycles
Operating settings
Multiple sensor measurements
Remaining Useful Life information

The raw dataset is not included in this repository.


## Model Performance Comparison

The following table summarizes the performance of the models evaluated
during the project.

| Model | Features | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---:|---:|---:|
| Random Forest | Original | 25.864 | 35.390 | 0.709 |
| Random Forest | Engineered | 24.467 | 34.424 | 0.725|
| XGBoost | Engineered | 23.36 | 32.946 | 0.748 |
| GRU | Engineered | 19.279 | 28.275 | 0.764 |
| stacked GRU | Engineered | 19.572 | 27.452 | 0.777 |
| Stacked LSTM | Engineered | 17.968 | 25.072 | 0.814 |


### Evaluation Metrics

- **MAE (Mean Absolute Error):** Average absolute difference between
  predicted and actual RUL. Lower values indicate smaller prediction errors.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors
  more strongly. Lower values indicate better performance.
- **R² (Coefficient of Determination):** Measures how much of the variance
  in RUL is explained by the model. Higher values indicate better fit.


  ### Results

The models were evaluated using the same evaluation methodology where
applicable. The results show how model architecture and feature
engineering affected RUL prediction performance throughout development.