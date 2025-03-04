"""
Improved Definition of Neural Network Models
"""
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Concatenate, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, LayerNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2


def get_lstm(units, with_features=False, feature_dim=None, dropout_rate=0.3, recurrent_dropout=0.2, l2_reg=0.001):
    """LSTM(Long Short-Term Memory)
    Build an improved LSTM Model with bidirectional layers, normalization, and regularization.

    # Arguments
        units: List[int], number of input, output and hidden units.
        with_features: Boolean, whether to include additional features.
        feature_dim: Integer, dimension of additional features.
        dropout_rate: Float, dropout rate for dense layers.
        recurrent_dropout: Float, dropout rate for recurrent layers.
        l2_reg: Float, L2 regularization factor.
    # Returns
        model: Model, nn model.
    """
    if with_features:
        # Input for time series data
        time_input = Input(shape=(units[0], 1), name='time_input')

        # First LSTM layer with bidirectional wrapper for better sequence learning
        lstm1 = Bidirectional(LSTM(units[1],
                                   return_sequences=True,
                                   recurrent_dropout=recurrent_dropout,
                                   kernel_regularizer=l2(l2_reg)))(time_input)
        lstm1 = LayerNormalization()(lstm1)

        # Second LSTM layer
        lstm2 = Bidirectional(LSTM(units[2],
                                   recurrent_dropout=recurrent_dropout,
                                   kernel_regularizer=l2(l2_reg)))(lstm1)
        lstm2 = LayerNormalization()(lstm2)

        # Input for additional features
        feature_input = Input(shape=(feature_dim,), name='feature_input')
        feature_norm = BatchNormalization()(feature_input)

        # Merge LSTM output with features
        merged = Concatenate()([lstm2, feature_norm])

        # Final layers with improved dropout and additional dense layer
        dropout1 = Dropout(dropout_rate)(merged)
        dense1 = Dense(units[2]//2, activation='relu',
                       kernel_regularizer=l2(l2_reg))(dropout1)
        dropout2 = Dropout(dropout_rate/2)(dense1)
        output = Dense(units[3], activation='sigmoid')(dropout2)

        # Create model with both inputs
        model = Model(inputs=[time_input, feature_input], outputs=output)
    else:
        # Sequential model
        model = Sequential()

        # First LSTM layer with bidirectional wrapper
        model.add(Bidirectional(LSTM(units[1],
                                     input_shape=(units[0], 1),
                                     return_sequences=True,
                                     recurrent_dropout=recurrent_dropout,
                                     kernel_regularizer=l2(l2_reg))))
        model.add(LayerNormalization())

        # Second LSTM layer
        model.add(Bidirectional(LSTM(units[2],
                                     recurrent_dropout=recurrent_dropout,
                                     kernel_regularizer=l2(l2_reg))))
        model.add(LayerNormalization())

        # Improved dropout and dense layers
        model.add(Dropout(dropout_rate))
        model.add(Dense(units[2]//2, activation='relu',
                  kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate/2))
        model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units, with_features=False, feature_dim=None, dropout_rate=0.3, recurrent_dropout=0.2, l2_reg=0.001):
    """GRU(Gated Recurrent Unit)
    Build an improved GRU Model with bidirectional layers, normalization, and regularization.

    # Arguments
        units: List[int], number of input, output and hidden units.
        with_features: Boolean, whether to include additional features.
        feature_dim: Integer, dimension of additional features.
        dropout_rate: Float, dropout rate for dense layers.
        recurrent_dropout: Float, dropout rate for recurrent layers.
        l2_reg: Float, L2 regularization factor.
    # Returns
        model: Model, nn model.
    """
    if with_features:
        # Input for time series data
        time_input = Input(shape=(units[0], 1), name='time_input')

        # First GRU layer with bidirectional wrapper
        gru1 = Bidirectional(GRU(units[1],
                                 return_sequences=True,
                                 recurrent_dropout=recurrent_dropout,
                                 kernel_regularizer=l2(l2_reg)))(time_input)
        gru1 = LayerNormalization()(gru1)

        # Second GRU layer
        gru2 = Bidirectional(GRU(units[2],
                                 recurrent_dropout=recurrent_dropout,
                                 kernel_regularizer=l2(l2_reg)))(gru1)
        gru2 = LayerNormalization()(gru2)

        # Input for additional features
        feature_input = Input(shape=(feature_dim,), name='feature_input')
        feature_norm = BatchNormalization()(feature_input)

        # Merge GRU output with features
        merged = Concatenate()([gru2, feature_norm])

        # Final layers with improved dropout and additional dense layer
        dropout1 = Dropout(dropout_rate)(merged)
        dense1 = Dense(units[2]//2, activation='relu',
                       kernel_regularizer=l2(l2_reg))(dropout1)
        dropout2 = Dropout(dropout_rate/2)(dense1)
        output = Dense(units[3], activation='sigmoid')(dropout2)

        # Create model with both inputs
        model = Model(inputs=[time_input, feature_input], outputs=output)
    else:
        # Sequential model
        model = Sequential()

        # First GRU layer with bidirectional wrapper
        model.add(Bidirectional(GRU(units[1],
                                    input_shape=(units[0], 1),
                                    return_sequences=True,
                                    recurrent_dropout=recurrent_dropout,
                                    kernel_regularizer=l2(l2_reg))))
        model.add(LayerNormalization())

        # Second GRU layer
        model.add(Bidirectional(GRU(units[2],
                                    recurrent_dropout=recurrent_dropout,
                                    kernel_regularizer=l2(l2_reg))))
        model.add(LayerNormalization())

        # Improved dropout and dense layers
        model.add(Dropout(dropout_rate))
        model.add(Dense(units[2]//2, activation='relu',
                  kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate/2))
        model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output, dropout_rate=0.3, l2_reg=0.001):
    """SAE(Auto-Encoders)
    Build an improved SAE Model with batch normalization and regularization.

    # Arguments
        inputs: int, number of input units.
        hidden: int, number of hidden units.
        output: int, number of output units.
        dropout_rate: Float, dropout rate.
        l2_reg: Float, L2 regularization factor.
    # Returns
        model: Model, nn model.
    """
    # Create an input layer
    input_layer = Input(shape=(inputs,))

    # Input normalization
    normalized = BatchNormalization()(input_layer)

    # Hidden layer with regularization
    hidden_layer = Dense(hidden, name='hidden',
                         kernel_regularizer=l2(l2_reg))(normalized)
    hidden_layer = BatchNormalization()(hidden_layer)
    hidden_layer = Activation('relu')(hidden_layer)
    dropout_layer = Dropout(dropout_rate)(hidden_layer)

    # Output layer
    output_layer = Dense(output, activation='sigmoid')(dropout_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def get_saes(layers, dropout_rates=[0.3, 0.3, 0.3, 0.2], l2_reg=0.001):
    """SAEs(Stacked Auto-Encoders)
    Build improved SAEs Model with batch normalization, better activation functions,
    and regularization.

    # Arguments
        layers: List[int], number of input, output and hidden units.
        dropout_rates: List[float], dropout rates for each layer.
        l2_reg: Float, L2 regularization factor.
    # Returns
        models: List[Model], List of SAE and SAEs.
    """
    # Individual autoencoders with improved architecture
    sae1 = _get_sae(layers[0], layers[1], layers[-1], dropout_rates[0], l2_reg)
    sae2 = _get_sae(layers[1], layers[2], layers[-1], dropout_rates[1], l2_reg)
    sae3 = _get_sae(layers[2], layers[3], layers[-1], dropout_rates[2], l2_reg)

    # Stacked autoencoder with improved architecture
    input_layer = Input(shape=(layers[0],))

    # Input normalization
    normalized = BatchNormalization()(input_layer)

    # First hidden layer
    hidden1 = Dense(layers[1], name='hidden1',
                    kernel_regularizer=l2(l2_reg))(normalized)
    hidden1 = BatchNormalization()(hidden1)
    hidden1 = Activation('relu')(hidden1)

    # Second hidden layer
    hidden2 = Dense(layers[2], name='hidden2',
                    kernel_regularizer=l2(l2_reg))(hidden1)
    hidden2 = BatchNormalization()(hidden2)
    hidden2 = Activation('relu')(hidden2)

    # Third hidden layer
    hidden3 = Dense(layers[3], name='hidden3',
                    kernel_regularizer=l2(l2_reg))(hidden2)
    hidden3 = BatchNormalization()(hidden3)
    hidden3 = Activation('relu')(hidden3)

    # Dropout and output
    dropout = Dropout(dropout_rates[3])(hidden3)
    output_layer = Dense(layers[4], activation='sigmoid')(dropout)

    # Create model
    saes = Model(inputs=input_layer, outputs=output_layer)

    models = [sae1, sae2, sae3, saes]

    return models
