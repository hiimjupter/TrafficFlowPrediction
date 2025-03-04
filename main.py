import math
import warnings
import numpy as np
import pandas as pd
import argparse
import pickle
import sys
import os
from pathlib import Path

import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras.models import load_model
from keras.losses import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from data.data import process_data

# Suppress warnings
warnings.filterwarnings("ignore")


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error

    Calculate the MAPE metric, handling zero values in the target data.

    Args:
        y_true: Array-like, true values
        y_pred: Array-like, predicted values

    Returns:
        float: MAPE value as a percentage
    """
    # Filter out zero values to avoid division by zero
    valid_indices = [i for i, x in enumerate(y_true) if x > 0]

    if not valid_indices:
        return 0.0

    filtered_y_true = [y_true[i] for i in valid_indices]
    filtered_y_pred = [y_pred[i] for i in valid_indices]

    # Calculate MAPE
    errors = [abs(true - pred) / true for true,
              pred in zip(filtered_y_true, filtered_y_pred)]
    mape = 100 * sum(errors) / len(errors)

    return mape


def calculate_all_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics

    Args:
        y_true: Array-like, true values
        y_pred: Array-like, predicted values

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    mape = calculate_mape(y_true, y_pred)
    evs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)

    return {
        "mape": mape,
        "evs": evs,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


def print_metrics(metrics_dict):
    """Print evaluation metrics in a formatted way

    Args:
        metrics_dict: Dictionary containing evaluation metrics
    """
    print(f"Explained Variance Score: {metrics_dict['evs']:.4f}")
    print(f"MAPE: {metrics_dict['mape']:.4f}%")
    print(f"MAE: {metrics_dict['mae']:.4f}")
    print(f"MSE: {metrics_dict['mse']:.4f}")
    print(f"RMSE: {metrics_dict['rmse']:.4f}")
    print(f"R²: {metrics_dict['r2']:.4f}")


def evaluate_model(y_true, y_pred):
    """Evaluate model performance and print metrics

    Args:
        y_true: Array-like, true values
        y_pred: Array-like, predicted values

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    metrics_dict = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics_dict)
    return metrics_dict


def plot_error_comparison(metrics_list, model_names):
    """Plot error metrics comparison across multiple models

    Args:
        metrics_list: List of dictionaries, each containing error metrics for a model
        model_names: List of model names corresponding to the metrics
    """
    if len(metrics_list) != len(model_names):
        raise ValueError(
            "Number of metric sets must match number of model names")

    metric_names = ["MAPE", "EVS", "MAE", "MSE", "RMSE", "R²"]
    metric_keys = ["mape", "evs", "mae", "mse", "rmse", "r2"]
    positions = range(len(model_names))

    # Extract metrics for each model
    metrics_by_type = {key: [metrics[key]
                             for metrics in metrics_list] for key in metric_keys}

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Plot each metric
    for i, (key, values) in enumerate(metrics_by_type.items()):
        ax = axes[i]
        bars = ax.bar(positions, values, width=0.6)

        # Add value annotations on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_title(metric_names[i])
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Set y-axis limits for EVS and R² to better show differences
        if key in ["evs", "r2"]:
            min_val = min(values) * 0.95
            min_val = max(min_val, 0)  # Don't go below 0
            ax.set_ylim(min_val, 1.02)

    plt.tight_layout()
    plt.savefig('images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_comparison(y_true, y_preds, model_names, save_path=None):
    """Plot the true data and predicted data for multiple models

    Args:
        y_true: Array-like, true values
        y_preds: List of arrays, predicted values for each model
        model_names: List of model names
        save_path: Optional path to save the plot
    """
    # Create time axis (96 15-minute periods in a day)
    start_date = '2006-10-26 00:00'
    x = pd.date_range(start_date, periods=96, freq='15min')

    plt.figure(figsize=(14, 8))

    # Plot true data with thicker line
    plt.plot(x, y_true, label='Ground Truth', linewidth=2.5, color='black')

    # Define a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    # Plot each model's predictions
    for i, (name, y_pred) in enumerate(zip(model_names, y_preds)):
        plt.plot(x, y_pred, label=name, linewidth=1.8,
                 alpha=0.8, color=colors[i])

    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Traffic Flow Prediction Comparison', fontsize=16)
    plt.xlabel('Time of Day', fontsize=14)
    plt.ylabel('Flow Volume', fontsize=14)

    # Format x-axis to show time
    date_format = mpl.dates.DateFormatter("%H:%M")
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()

    # Add a light horizontal grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def load_neural_network_models(model_paths):
    """Load all neural network models

    Args:
        model_paths: List of paths to the model files

    Returns:
        list: Loaded models
    """
    custom_objects = {
        'mse': mean_squared_error,
        'mape': mean_absolute_percentage_error,
        'mae': mean_absolute_error
    }

    models = []
    for path in model_paths:
        try:
            model = load_model(path, custom_objects=custom_objects)
            models.append(model)
        except Exception as e:
            print(f"Error loading model {path}: {e}")
            models.append(None)

    return models


def load_ml_models(model_paths):
    """Load classical machine learning models from pickle files

    Args:
        model_paths: List of paths to the model files

    Returns:
        list: Loaded models
    """
    models = []
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                models.append(pickle.load(f))
        except Exception as e:
            print(f"Error loading model {path}: {e}")
            models.append(None)

    return models


def predict_with_model(model, X_test, X_test_time, X_test_features, scaler, model_type, name):
    """Generate predictions using the specified model

    Args:
        model: The model to use for prediction
        X_test: Full test dataset
        X_test_time: Time series portion of test data
        X_test_features: Feature portion of test data
        scaler: Data scaler for inverse transformation
        model_type: Type of the model ('nn' or 'ml')
        name: Name of the model

    Returns:
        array: Predictions after inverse scaling
    """
    if model is None:
        print(f"Skipping {name} as model failed to load")
        return None

    print(f"Generating predictions for {name}...")

    if model_type == 'nn':
        if name == 'SAEs':
            # SAE takes flattened input
            predicted = model.predict(X_test)
        else:
            # For LSTM and GRU, check if they expect separate inputs
            if isinstance(model.input, list):
                # Model expects separate time series and feature inputs
                X_test_time_reshaped = np.reshape(
                    X_test_time, (X_test_time.shape[0], X_test_time.shape[1], 1))
                predicted = model.predict(
                    [X_test_time_reshaped, X_test_features])
            else:
                # Model expects only time series input
                X_test_reshaped = np.reshape(
                    X_test_time, (X_test_time.shape[0], X_test_time.shape[1], 1))
                predicted = model.predict(X_test_reshaped)
    else:
        # Classical ML models use the full feature set
        predicted = model.predict(X_test)

    # Inverse transform predictions to original scale
    predicted = scaler.inverse_transform(
        predicted.reshape(-1, 1)).reshape(1, -1)[0]

    return predicted


def ensure_directory_exists(path):
    """Ensure that a directory exists, creating it if needed

    Args:
        path: Path to the directory
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    """Main function to run the evaluation pipeline"""
    # Set up directories and file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, 'data', 'splitted_scats')
    model_folder = os.path.join(base_dir, 'model')
    image_folder = os.path.join(base_dir, 'images')

    # Ensure necessary directories exist
    ensure_directory_exists(image_folder)

    # Define neural network models
    nn_model_names = ['LSTM', 'GRU', 'SAEs']
    nn_model_files = [os.path.join(
        model_folder, f"{name.lower()}.h5") for name in nn_model_names]

    # Define classical ML models
    ml_model_names = ['Random Forest', 'XGBoost', 'CatBoost']
    ml_model_files = [os.path.join(model_folder, f"{name.lower().replace(' ', '_')}.pkl")
                      for name in ['random_forest', 'xgboost', 'catboost']]

    # Load all models
    print("Loading neural network models...")
    nn_models = load_neural_network_models(nn_model_files)

    print("Loading machine learning models...")
    ml_models = load_ml_models(ml_model_files)

    # Plot neural network model architectures
    for name, model in zip(nn_model_names, nn_models):
        if model is not None:
            try:
                plot_model(model, to_file=os.path.join(image_folder, f"{name.lower()}_architecture.png"),
                           show_shapes=True, show_layer_names=True, expand_nested=True)
                print(f"Saved {name} architecture diagram")
            except Exception as e:
                print(f"Could not plot {name} architecture: {e}")

    # Set up data
    lag = 12
    train_file = os.path.join(
        data_folder, '0970_HIGH STREET_RD E of WARRIGAL_RD_train.csv')
    test_file = os.path.join(
        data_folder, '0970_HIGH STREET_RD E of WARRIGAL_RD_test.csv')

    # Process data
    print("Processing data...")
    X_train, y_train, X_test, y_test, scaler = process_data(
        train_file, test_file, lag)

    # Inverse transform y_test to original scale
    y_test_original = scaler.inverse_transform(
        y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # Separate time series and feature portions
    # Assuming first 'lag' columns are time series data
    X_test_time = X_test[:, :lag]
    X_test_features = X_test[:, lag:] if X_test.shape[1] > lag else None

    # Prepare for predictions
    all_models = nn_models + ml_models
    all_model_names = nn_model_names + ml_model_names
    all_model_types = ['nn'] * len(nn_models) + ['ml'] * len(ml_models)

    # Generate predictions and evaluate models
    predictions = []
    metrics_list = []

    print("\n" + "="*50)
    print("Starting model evaluation")
    print("="*50)

    for model, name, model_type in zip(all_models, all_model_names, all_model_types):
        if model is None:
            predictions.append(None)
            continue

        print(f"\nEvaluating {name}...")
        predicted = predict_with_model(model, X_test, X_test_time, X_test_features,
                                       scaler, model_type, name)

        if predicted is not None:
            print("Model evaluation metrics:")
            metrics = evaluate_model(y_test_original, predicted)
            metrics_list.append(metrics)
            # Keep first day for visualization
            predictions.append(predicted[:96])
        else:
            predictions.append(None)

    # Filter out None values
    valid_predictions = [p for p in predictions if p is not None]
    valid_model_names = [name for name, p in zip(
        all_model_names, predictions) if p is not None]

    # Visualize results
    if valid_predictions:
        print("\nPlotting prediction comparison...")
        plot_prediction_comparison(y_test_original[:96], valid_predictions, valid_model_names,
                                   save_path=os.path.join(image_folder, 'prediction_comparison.png'))

        print("Plotting error metrics comparison...")
        plot_error_comparison(metrics_list, valid_model_names)
    else:
        print("No valid predictions to visualize.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
