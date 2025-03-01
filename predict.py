import warnings
import argparse
import sys
import math
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
import joblib
from catboost import CatBoostRegressor


warnings.filterwarnings("ignore")


def flow_prediction(scats_num, location, time):
    data_folder = '/Users/jupternguyen/Projects/TrafficFlowPrediction/data/splitted_scats'

    # Load CatBoost model from .pkl file
    model_path = f'model/random_forest/{scats_num}/{location}.pkl'
    catboost_model = joblib.load(model_path)

    # Prepare the input data for prediction
    time = pd.to_datetime(time)
    lag = 12
    test_file = f'{data_folder}/{scats_num}_{location}_test.csv'
    _, _, X_test, _, scaler = process_data(test_file, test_file, lag)

    # Find the index of the given time
    time_index = (time - pd.to_datetime('2006-10-26 00:00')).seconds // 900

    # Reshape the data for the CatBoost model
    X_test_cb = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Predict the traffic flow
    predicted = catboost_model.predict(X_test_cb)
    predicted = scaler.inverse_transform(
        predicted.reshape(-1, 1)).reshape(1, -1)[0]

    # Print the predicted traffic flow at the given time
    print('Predicted traffic flow:', round(predicted[time_index]))
    return round(predicted[time_index])


def speed_calculation(scats_num, location, time):
    flow = flow_prediction(scats_num, location, time)

    # Calculate the speed
    lambda_decay = 0.01  # Decay rate
    F_base = 275  # Base flow, below which speed is 60 km/h (uncongested)
    excess_flow = max(flow - F_base, 0)
    v_eff = 60 * math.exp(-lambda_decay * excess_flow)

    # Print the effective speed
    print('Effective speed:', round(v_eff))

    return round(v_eff)


def convert_time(hours):
    # Convert hours to minutes
    minutes = hours * 60
    # Extract whole minutes
    whole_minutes = int(minutes)
    # Calculate remaining seconds
    seconds = (minutes - whole_minutes) * 60
    return whole_minutes, int(seconds)


def compute_eta(location_lst, distance_lst, time):
    total_time = 0
    for i in range(len(location_lst) - 1):
        scats_num = location_lst[i][:4]
        location = location_lst[i][5:]
        speed = speed_calculation(scats_num, location, time)
        distance = distance_lst[i]
        time = distance / speed
        total_time += time

    # Add 30s for every intersection
    total_time += (30 / 3600) * (len(location_lst) - 1)  # Convert 30s to hours

    minutes, seconds = convert_time(total_time)
    print('Estimated time of arrival:',
          # Print estimated time of arrival in minutes and seconds
          f'{minutes} minutes and {seconds} seconds')

    return total_time  # Return total time in hours


if __name__ == '__main__':
    # Example of going from A to C with B in between
    compute_eta(['2825_BURKE_RD S of EASTERN_FWY', '4032_BELMORE_RD E of BURKE_RD',
                 '0970_WARRIGAL_RD N of HIGH STREET_RD', '3127_CANTERBURY_RD E of BALWYN_RD'], [3.2, 4, 8], '2025-02-26 10:15')
