import warnings
import argparse
import sys
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model


warnings.filterwarnings("ignore")


def predict(argv):
    data_folder = '/Users/jupternguyen/Projects/TrafficFlowPrediction/data/splitted_scats'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scats_num",
        required=True,
        help="SCATS number, 4 digits.")
    parser.add_argument(
        "--location",
        required=True,
        help="Location.")
    parser.add_argument(
        "--time",
        required=True,
        help='Time of day in format "YYYY-MM-DD HH:MM", e.g. "2006-10-22 00:00", 15-minute intervals')
    args = parser.parse_args()

    # Load NN model
    saes = load_model(
        'model/saes/{0}/{1}.h5'.format(args.scats_num, args.location))

    # Prepare the input data for prediction
    time = pd.to_datetime(args.time)
    lag = 12
    test_file = f'{data_folder}/{args.scats_num}_{args.location}_test.csv'
    _, _, X_test, _, scaler = process_data(test_file, test_file, lag)

    # Find the index of the given time
    time_index = (time - pd.to_datetime('2006-10-26 00:00')).seconds // 900

    # Reshape the data for the SAEs model
    X_test_nn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    # Predict the traffic flow
    predicted = saes.predict(X_test_nn)
    predicted = scaler.inverse_transform(
        predicted.reshape(-1, 1)).reshape(1, -1)[0]

    # Print the predicted traffic flow at the given time
    print(
        f"Predicted traffic flow at {args.scats_num}_{args.location}, at {args.time}: {round(predicted[time_index])}")


if __name__ == '__main__':
    predict(sys.argv[1:])
