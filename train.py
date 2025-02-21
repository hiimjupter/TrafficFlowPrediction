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
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, scats, name, config):
    """Train a neural network model."""
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    save_location = 'model/{name}/{scats}'.format(name=name, scats=scats)

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    model.save(save_location + '/model.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(save_location + '/model_loss.csv', encoding='utf-8', index=False)


def train_ml_model(model, X_train, y_train, scats, name, config):
    """Train a classical ML model such as RandomForest, XGBoost, or CatBoost."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    mape = mean_absolute_percentage_error(y_train, y_pred)

    save_location = 'model/{name}/{scats}'.format(name=name, scats=scats)

    # Save model using pickle
    with open(save_location + '/model' + '.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Training completed for {name} with MSE: {mse} and MAPE: {mape}")


def train_seas(models, X_train, y_train, scats, name, config):
    """Train the SAEs model."""
    temp = X_train
    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(inputs=p.input,
                                       outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, scats, name, config)


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
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, scats_num, args.model, config)
    elif args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, scats_num, args.model, config)
    elif args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([12, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, scats_num, args.model, config)
    elif args.model == 'random_forest':
        m = RandomForestRegressor(n_estimators=100, random_state=42)
        train_ml_model(m, X_train, y_train, scats_num, args.model, config)
    elif args.model == 'xgboost':
        m = XGBRegressor(n_estimators=100, learning_rate=0.1,
                         max_depth=3, random_state=42)
        train_ml_model(m, X_train, y_train, scats_num, args.model, config)
    elif args.model == 'catboost':
        m = CatBoostRegressor(iterations=100, learning_rate=0.1,
                              depth=6, random_state=42, verbose=0)
        train_ml_model(m, X_train, y_train, scats_num, args.model, config)
    else:
        print(f"Model {args.model} is not supported.")
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv)
