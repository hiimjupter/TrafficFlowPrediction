import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle
import os
import glob
from tqdm import tqdm

warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """Train a neural network model."""
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)


def train_ml_model(model, X_train, y_train, name, config):
    """Train a classical ML model such as RandomForest, XGBoost, or CatBoost."""

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    mape = mean_absolute_percentage_error(y_train, y_pred)

    # Save model and metrics
    pd.DataFrame({'MSE': [mse], 'MAPE': [mape]}).to_csv(
        'model/' + name + '_metrics.csv', encoding='utf-8', index=False)

    # Save model as pickle file
    with open('model/' + name + '.pkl', 'wb') as f:
        pickle.dump(model, f)

    print(f"Training completed for {name} with MSE: {mse} and MAPE: {mape}")
    print(f"Model saved to model/{name}.pkl")


def train_saes(models, X_train, y_train, name, config):
    """Train the SAEs model."""
    temp = X_train

    # First, ensure all models are compiled
    for i in range(len(models) - 1):
        models[i].compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

    # Train each autoencoder separately
    for i in range(len(models) - 1):
        print(f"Training autoencoder {i+1}")

        # Train current autoencoder
        m = models[i]
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        # Get the output of the hidden layer to feed into the next autoencoder
        # For Sequential models, we need to create a new model to extract hidden layer outputs
        hidden_output = Model(inputs=m.input,
                              outputs=m.get_layer('hidden').output)
        temp = hidden_output.predict(temp)

        models[i] = m

    # Stack all trained autoencoders into the final SAES model
    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    data_folder = '/Users/jupternguyen/Projects/TrafficFlowPrediction/data/splitted_scats'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    parser.add_argument(
        "--scats_num",
        required=True,
        help="SCATS number.")
    parser.add_argument(
        "--location",
        required=True,
        help="Location.")
    args = parser.parse_args()

    # Lag for the model
    lag = 12
    # Training configuration
    config = {"batch": 256, "epochs": 600}

    scats_num = args.scats_num
    location = args.location
    train = f'{data_folder}/{scats_num}_{location}_train.csv'
    test = f'{data_folder}/{scats_num}_{location}_test.csv'

    X_train, y_train, _, _, _ = process_data(train, test, lag)

    if args.model == 'lstm':
        # For LSTM models, need to reshape the lag features for sequential input
        # Extract lag features (previous time steps) and other features
        lag_features = X_train[:, :lag]
        other_features = X_train[:, lag:]

        # Reshape lag features into 3D tensor for LSTM: (samples, time steps, features)
        # This creates proper sequential data representation
        lag_features_reshaped = np.reshape(
            lag_features, (lag_features.shape[0], lag_features.shape[1], 1))

        print(f"Lag features shape: {lag_features_reshaped.shape}")
        print(f"Other features shape: {other_features.shape}")

        # Create LSTM model with both sequential and non-sequential inputs
        m = model.get_lstm([lag, 64, 64, 1], with_features=True,
                           feature_dim=other_features.shape[1])

        # Train the model with both types of inputs
        train_model(m, [lag_features_reshaped, other_features],
                    y_train, args.model, config)

    elif args.model == 'gru':
        # Similar approach for GRU
        lag_features = X_train[:, :lag]
        other_features = X_train[:, lag:]
        lag_features_reshaped = np.reshape(
            lag_features, (lag_features.shape[0], lag_features.shape[1], 1))

        m = model.get_gru([lag, 64, 64, 1], with_features=True,
                          feature_dim=other_features.shape[1])
        train_model(m, [lag_features_reshaped, other_features],
                    y_train, args.model, config)

    # For ML models, you can use X_train directly as they can handle tabular data
    elif args.model in ['random_forest', 'xgboost', 'catboost']:
        if args.model == 'random_forest':
            m = RandomForestRegressor(n_estimators=100, random_state=42)
        elif args.model == 'xgboost':
            m = XGBRegressor(n_estimators=100, learning_rate=0.1,
                             max_depth=3, random_state=42)
        elif args.model == 'catboost':
            m = CatBoostRegressor(iterations=100, learning_rate=0.1,
                                  depth=6, random_state=42, verbose=0)
        train_ml_model(m, X_train, y_train, args.model, config)

    # For SAES, you may need to adjust the architecture
    elif args.model == 'saes':
        input_dim = X_train.shape[1]  # Use full input dimension
        m = model.get_saes([input_dim, 400, 400, 400, 1])
        train_saes(m, X_train, y_train, args.model, config)

    else:
        print(f"Model {args.model} is not supported.")
        sys.exit(1)


def train_all_random_forest():
    """Train CatBoost models for all SCATS sites in the dataset."""

    data_folder = '/Users/jupternguyen/Projects/TrafficFlowPrediction/data/splitted_scats'
    output_folder = 'model/random_forest'

    # Get all training files
    train_files = glob.glob(f'{data_folder}/*_train.csv')

    # Configuration
    lag = 12
    config = {"batch": 256, "epochs": 600}

    # Create base output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for train_file in tqdm(train_files, desc="Training Random_Forest models"):
        # Extract SCATS number and location from filename
        filename = os.path.basename(train_file)
        parts = filename.replace('_train.csv', '').split('_')
        scats_num = parts[0]
        # Extract location by removing the SCATS number and '_train.csv' parts
        location = filename.replace(
            f'{scats_num}_', '').replace('_train.csv', '')

        # Create directory structure
        model_dir = f'{output_folder}/{scats_num}'
        os.makedirs(model_dir, exist_ok=True)

        # Get test file
        test_file = train_file.replace('_train.csv', '_test.csv')

        # Process data
        X_train, y_train, _, _, _ = process_data(train_file, test_file, lag)

        # Train CatBoost model
        print(f"Training model for {scats_num}_{location}...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        mape = mean_absolute_percentage_error(y_train, y_pred)

        # Save metrics
        pd.DataFrame({'MSE': [mse], 'MAPE': [mape]}).to_csv(
            f'{model_dir}/{location}_metrics.csv', encoding='utf-8', index=False)

        # Save model
        output_path = f'{model_dir}/{location}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved to {output_path}")

    print(f"All Random_Forest models trained and saved to {output_folder}")


if __name__ == '__main__':
    # main(sys.argv)
    train_all_random_forest()
