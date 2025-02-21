import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import catboost
from catboost import CatBoostRegressor
import pickle  # added import

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
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2006-10-25 19:00'
    num_periods = len(y_true)
    x = pd.date_range(d, periods=num_periods, freq='15min')

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


def main():
    data_folder = '/Users/jupternguyen/Projects/TrafficFlowPrediction/data/splitted_scats'
    # Load NN models
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    models_nn = [lstm, gru, saes]
    names_nn = ['LSTM', 'GRU', 'SAEs']

    # Load classical ML models from pickle files instead of creating new ones
    names_ml = ['random_forest', 'xgboost', 'catboost']
    models_ml = []
    for name in names_ml:
        with open('model/' + name + '.pkl', 'rb') as f:
            models_ml.append(pickle.load(f))
    display_names_ml = ['Random Forest', 'XGBoost', 'CatBoost']

    # Set up the lag and data
    lag = 12
    train = f'{data_folder}/2000_BURWOOD_HWY E of WARRIGAL_RD_train.csv'
    test = f'{data_folder}/2000_BURWOOD_HWY E of WARRIGAL_RD_test.csv'
    _, _, X_test, y_test, scaler = process_data(train, test, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    # Prepare the list to store predictions and model names
    y_preds = []
    all_names = names_nn + display_names_ml
    all_models = models_nn + models_ml

    # First process the neural network models
    for name, model in zip(names_nn, models_nn):
        if name == 'SAEs':
            X_test_nn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test_nn = np.reshape(
                X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test_nn)
        predicted = scaler.inverse_transform(
            predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:960])
        print(name)
        eva_regress(y_test, predicted)

    # Process the classical machine learning models (loaded from pkl files)
    for name, model in zip(display_names_ml, models_ml):
        # use loaded model without re-training
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(
            predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:960])
        print(name)
        eva_regress(y_test, predicted)

    # Plot the results
    plot_results(y_test[:960], y_preds, all_names)


if __name__ == '__main__':
    main()
