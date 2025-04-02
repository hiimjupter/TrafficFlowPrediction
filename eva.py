import os
import glob
import numpy as np
import pickle
import pandas as pd
from keras.models import load_model
from data.data import process_data
from MainRoadData.data import process_newdata
import math
import warnings
from data.data import process_data
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import catboost
from catboost import CatBoostRegressor
import argparse
import sys

warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    Evaluate the predicted result.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    mtx = {
        "mape": mape,
        "evs": vs,
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2
    }

    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

    return mtx


def plot_error(mtx):
    """Plot error metrics for each model.

    # Arguments
        mtx: List of dictionaries, each containing error metrics for a model.
    """

    labels = ["MAPE", "EVS", "MAE", "MSE", "RMSE", "R2"]
    model_names = ['LSTM', 'GRU', 'SAEs',
                'Random Forest', 'XGBoost', 'CatBoost']
    positions = range(len(model_names))

    # Initialize lists to store error metrics
    mape, evs, mae, mse, rmse, r2 = [], [], [], [], [], []

    # Extract error metrics for each model
    for metrics in mtx:
        mape.append(metrics["mape"])
        evs.append(metrics["evs"])
        mae.append(metrics["mae"])
        mse.append(metrics["mse"])
        rmse.append(metrics["rmse"])
        r2.append(metrics["r2"])

    error_measurements = [mape, evs, mae, mse, rmse, r2]

    # Plot each error metric
    plt.figure(figsize=(12, 10))
    for i, em in enumerate(error_measurements):
        plt.subplot(3, 2, i + 1)
        plt.bar(positions, em, width=0.5)
        plt.xticks(positions, model_names, rotation=45)
        plt.title(labels[i])
        plt.tight_layout()
        if labels[i] in ["EVS", "R2"]:
            plt.ylim(0.9, 1)

    plt.show()


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2022-12-20 01:00'

    # 1440 minutes in a day = 96 x 15 minute periods
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def apply_predictions_all_files(data_folder, model_folder):
    """Apply predictions on all train-test file pairs and compute mean evaluation metrics per model."""

    # Load NN models
    try:
        lstm = load_model(os.path.join(model_folder, 'lstm.h5'))
        gru = load_model(os.path.join(model_folder, 'gru.h5'))
        saes = load_model(os.path.join(model_folder, 'saes.h5'))
        models_nn = [lstm, gru, saes]
        names_nn = ['LSTM', 'GRU', 'SAEs']
    except Exception as e:
        print(f"Error loading neural network models: {e}")
        return

    # Load classical ML models
    names_ml = ['random_forest', 'xgboost', 'catboost']
    models_ml = []
    try:
        for name in names_ml:
            with open(os.path.join(model_folder, f'{name}.pkl'), 'rb') as f:
                models_ml.append(pickle.load(f))
        display_names_ml = ['Random Forest', 'XGBoost', 'CatBoost']
    except Exception as e:
        print(f"Error loading classical ML models: {e}")
        return

    all_results = []
    train_files = glob.glob(os.path.join(data_folder, '*_train.csv'))
    
    # Initialize result storage for each model
    model_results = {
        'LSTM': [],
        'GRU': [],
        'SAEs': [],
        'Random Forest': [],
        'XGBoost': [],
        'CatBoost': []
    }
    
    for train_file in train_files:
        test_file = train_file.replace('_train.csv', '_test.csv')
        
        if not os.path.exists(test_file):
            print(f"Skipping {train_file} as the corresponding test file {test_file} does not exist.")
            continue  # Skip if test file is missing

        # Print information about the processed files
        print(f"This is the result of {os.path.basename(train_file)} and {os.path.basename(test_file)}")
        
        # Process data
        try:
            lag = 12
            _, _, X_test, y_test, scaler = process_newdata(train_file, test_file, lag)
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
        except Exception as e:
            print(f"Error processing data for {train_file} and {test_file}: {e}")
            continue  # Skip this pair if an error occurs during data processing
        
        # Predict using NN models
        for name, model in zip(names_nn, models_nn):
            try:
                X_test_nn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1])) if name == 'SAEs' \
                            else np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predicted = model.predict(X_test_nn)
                predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
                
                # Evaluate and store results
                metrics = eva_regress(y_test, predicted)
                model_results[name].append(metrics)
            except Exception as e:
                print(f"Error with {name} model for {train_file}: {e}")
                continue  # Skip if an error occurs during prediction or evaluation
        
        # Predict using ML models
        for name, model in zip(display_names_ml, models_ml):
            try:
                predicted = model.predict(X_test)
                predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
                
                # Evaluate and store results
                metrics = eva_regress(y_test, predicted)
                model_results[name].append(metrics)
            except Exception as e:
                print(f"Error with {name} model for {train_file}: {e}")
                continue  # Skip if an error occurs during prediction or evaluation
    
    # Compute mean metrics for each model
    mean_results_per_model = {}
    for model_name, results in model_results.items():
        if results:  # Check if the model has any results
            mean_results_per_model[model_name] = {key: np.mean([res[key] for res in results]) for key in results[0]}
    
    # Print mean metrics for each model
    print("\nMean Evaluation Metrics per Model:")
    for model_name, metrics in mean_results_per_model.items():
        print(f"\n{model_name}:")
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
    
    print("Prediction completed for all files.")
    return mean_results_per_model



# Example usage
data_folder = '/Users/LINH KHANH/COS30018/TrafficFlowPrediction/MainRoadData/Splitted_MLink'
model_folder = 'model'
apply_predictions_all_files(data_folder, model_folder)
