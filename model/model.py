"""
Definition of NN model
"""
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Concatenate
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Sequential, Model


def get_lstm(units, with_features=False, feature_dim=None):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List[int], number of input, output and hidden units.
        with_features: Boolean, whether to include additional features.
        feature_dim: Integer, dimension of additional features.
    # Returns
        model: Model, nn model.
    """
    if with_features:
        # Input for time series data
        time_input = Input(shape=(units[0], 1), name='time_input')
        lstm1 = LSTM(units[1], return_sequences=True)(time_input)
        lstm2 = LSTM(units[2])(lstm1)

        # Input for additional features
        feature_input = Input(shape=(feature_dim,), name='feature_input')

        # Merge LSTM output with features
        merged = Concatenate()([lstm2, feature_input])

        # Final layers
        dropout = Dropout(0.2)(merged)
        output = Dense(units[3], activation='sigmoid')(dropout)

        # Create model with both inputs
        model = Model(inputs=[time_input, feature_input], outputs=output)
    else:
        model = Sequential()
        model.add(LSTM(units[1], input_shape=(
            units[0], 1), return_sequences=True))
        model.add(LSTM(units[2]))
        model.add(Dropout(0.2))
        model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units, with_features=False, feature_dim=None):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List[int], number of input, output and hidden units.
        with_features: Boolean, whether to include additional features.
        feature_dim: Integer, dimension of additional features.
    # Returns
        model: Model, nn model.
    """
    if with_features:
        # Input for time series data
        time_input = Input(shape=(units[0], 1), name='time_input')
        gru1 = GRU(units[1], return_sequences=True)(time_input)
        gru2 = GRU(units[2])(gru1)

        # Input for additional features
        feature_input = Input(shape=(feature_dim,), name='feature_input')

        # Merge GRU output with features
        merged = Concatenate()([gru2, feature_input])

        # Final layers
        dropout = Dropout(0.2)(merged)
        output = Dense(units[3], activation='sigmoid')(dropout)

        # Create model with both inputs
        model = Model(inputs=[time_input, feature_input], outputs=output)
    else:
        model = Sequential()
        model.add(GRU(units[1], input_shape=(
            units[0], 1), return_sequences=True))
        model.add(GRU(units[2]))
        model.add(Dropout(0.2))
        model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: int, number of input units.
        hidden: int, number of hidden units.
        output: int, number of output units.
    # Returns
        model: Model, nn model.
    """
    # Create an input layer explicitly to ensure the model has a defined input
    input_layer = Input(shape=(inputs,))
    hidden_layer = Dense(hidden, name='hidden')(input_layer)
    hidden_layer = Activation('sigmoid')(hidden_layer)
    dropout_layer = Dropout(0.2)(hidden_layer)
    output_layer = Dense(output, activation='sigmoid')(dropout_layer)

    # Create a Model instead of Sequential to have clearly defined inputs/outputs
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List[int], number of input, output and hidden units.
    # Returns
        models: List[Model], List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    input_layer = Input(shape=(layers[0],))
    hidden1 = Dense(layers[1], name='hidden1',
                    activation='sigmoid')(input_layer)
    hidden2 = Dense(layers[2], name='hidden2', activation='sigmoid')(hidden1)
    hidden3 = Dense(layers[3], name='hidden3', activation='sigmoid')(hidden2)
    dropout = Dropout(0.2)(hidden3)
    output_layer = Dense(layers[4], activation='sigmoid')(dropout)

    saes = Model(inputs=input_layer, outputs=output_layer)

    models = [sae1, sae2, sae3, saes]

    return models
