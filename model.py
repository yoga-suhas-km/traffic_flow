"""
Defination of NN model
"""
import sys
from sklearn.ensemble import RandomForestRegressor
from keras.layers import Dense, Dropout, Activation, Conv1D, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import AdaBoostRegressor

def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1)))
    #model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1)))
    #model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models

def get_cnn_lstm(layers):
    activation = 'relu'

    model = Sequential()

    model.add(Conv1D(layers[1], strides=1, input_shape=(layers[0], 1), activation=activation, kernel_size=1, padding='valid'))
    model.add(Conv1D(layers[1], strides=1, activation=activation, kernel_size=1, padding='valid'))
    model.add(LSTM(layers[1],return_sequences=True))
    model.add(Dense(layers[1], activation=activation))
    model.add(Dense(layers[1], activation=activation))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(layers[2], activation=activation))    
    
    return model
    
def get_rf():
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    return regressor

def get_cnn_lstm_en(layers, name):
    activation = 'relu'

    model = Sequential()

    model.add(Conv1D(layers[1], strides=1, input_shape=(layers[0], 1), activation=activation, kernel_size=1, padding='valid',name=name))
    model.add(Conv1D(layers[1], strides=1, activation=activation, kernel_size=1, padding='valid'))
    model.add(LSTM(layers[1],return_sequences=True))
    model.add(Dense(layers[1], activation=activation))
    model.add(Dense(layers[1], activation=activation))
    model.add(Dense(layers[1], activation=activation))
    model.add(Flatten())
    model.add(Dropout(0.2))     
    model.add(Dense(layers[3], activation=activation))    

    return model    
    

def get_gru_en(units, name):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    #model.add(GRU(units[1],return_sequences=True, name=name, input_shape=(units[0], 1)))
    model.add(GRU(units[1], name=name, input_shape=(units[0], 1)))
    #model.add(GRU(units[1]))
    #model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='relu'))

    return model    

def get_lstm_en(units, name):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    #model.add(LSTM(units[1],return_sequences=True, name=name,input_shape=(units[0], 1)))
    model.add(LSTM(units[1], name=name,input_shape=(units[0], 1)))
    #model.add(LSTM(units[1]))
    #model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model    

def get_ensemble_en_1(units):
    model_1 = get_lstm_en(units, "hidden")
    model_2 = get_cnn_lstm_en(units, "hidden1")

    models = [model_1, model_2]

    return models

def get_ensemble_en_2(units):
    model_1 = get_gru_en(units, "hidden")
    model_2 = get_cnn_lstm_en(units, "hidden1")

    models = [model_1, model_2]

    return models

def get_ensemble_en_3(units):
    model_1 = get_gru_en(units, "hidden")
    model_2 = get_lstm_en(units, "hidden1")

    models = [model_1, model_2]

    return models