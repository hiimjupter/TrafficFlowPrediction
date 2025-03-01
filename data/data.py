"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def split_data(scats_num, location, split_ratio):
    # read data
    df = pd.read_csv(input_dir + '/' + str(scats_num) +
                     '_' + location + '.csv')
    # Calculate index for splitting data based on the specified split_ratio
    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index]  # Select rows for training data
    test_df = df.iloc[split_index:]   # Select rows for testing data
    # Save data to the output directory
    train_df.to_csv(output_dir + '/' + str(scats_num) + '_' +
                    location + '_train.csv', index=False)
    test_df.to_csv(output_dir + '/' + str(scats_num) +
                   '_' + location + '_test.csv', index=False)


def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: MinMax Scaler.
    """
    target_attr = 'Lane 1 Flow (Veh/15 Minutes)'
    feature_cols = ['Day', 'Hour', 'Hour-Sin',
                    'Hour-Cos', 'Lag-15min', 'Lag-30min']

    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # Scale the target variable
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(
        df1[target_attr].values.reshape(-1, 1))
    flow1 = scaler.transform(
        df1[target_attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(
        df2[target_attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # Extract features
    features_train = df1[feature_cols].values
    features_test = df2[feature_cols].values

    # Feature scaling for numerical features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    features_train_scaled = feature_scaler.fit_transform(features_train)
    features_test_scaled = feature_scaler.transform(features_test)

    train_data, test_data = [], []

    # Create sequences for time series prediction
    for i in range(lags, len(flow1)):
        # Combine flow lags with other features
        train_sample = np.concatenate([
            flow1[i-lags:i],  # Past flow values
            features_train_scaled[i]  # Current features
        ])
        # Append current flow as target
        train_data.append(np.append(train_sample, flow1[i]))

    for i in range(lags, len(flow2)):
        test_sample = np.concatenate([
            flow2[i-lags:i],  # Past flow values
            features_test_scaled[i]  # Current features
        ])
        # Append current flow as target
        test_data.append(np.append(test_sample, flow2[i]))

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    np.random.shuffle(train_data)

    # Split features and target
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    return X_train, y_train, X_test, y_test, scaler
