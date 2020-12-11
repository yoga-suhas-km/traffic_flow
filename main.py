"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import sys
import warnings
import argparse
import math
import warnings
import numpy as np
import pandas as pd
from process_data import load_data
#from keras.models import load_model
import _pickle as cPickle
import tensorflow as tf 

from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
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
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
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
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2015-10-04 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

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


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="pems",
        help="data to use")
    args = parser.parse_args()
    
    if args.data == "pems":
        lstm = tf.keras.models.load_model('model_pems/lstm.h5')
       
        gru = tf.keras.models.load_model('model_pems/gru.h5')

        saes = tf.keras.models.load_model('model_pems/saes.h5')
        
        cnn_lstm = tf.keras.models.load_model('model_pems/cnn_lstm.h5')
        
        with open('model_pems/rf.h5', 'rb') as f:
            rf = cPickle.load(f)    
        
        en_1 = tf.keras.models.load_model('model_pems/en_1.h5')
        
        en_2 = tf.keras.models.load_model('model_pems/en_2.h5')
        
        en_3 = tf.keras.models.load_model('model_pems/en_3.h5')
    
    elif args.data == "nyc":

        lstm = tf.keras.models.load_model('model_nyc/lstm.h5')
       
        gru = tf.keras.models.load_model('model_nyc/gru.h5')

        saes = tf.keras.models.load_model('model_nyc/saes.h5')
        
        cnn_lstm = tf.keras.models.load_model('model_nyc/cnn_lstm.h5')
        
        with open('model_nyc/rf.h5', 'rb') as f:
            rf = cPickle.load(f)    
        
        en_1 = tf.keras.models.load_model('model_nyc/en_1.h5')
        
        en_2 = tf.keras.models.load_model('model_nyc/en_2.h5')
        
        en_3 = tf.keras.models.load_model('model_nyc/en_3.h5')


    models = [lstm, gru, saes, cnn_lstm, rf, en_1, en_2, en_3]
    names = ['LSTM', 'GRU', 'SAEs', 'CNN_LSTM', 'rf', 'EN_1', 'EN_2', 'EN_3']


    if args.data == "pems":
        X_train, X_test, y_train, y_test, scaler = load_data(data = "PEMS traffic prediction", force_download = False)
    elif args.data == "nyc":
        X_train, X_test, y_train, y_test, scaler = load_data(data = "nyc_bike_dataset", force_download = False)

    rf_bk = X_test
    
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        elif name == 'LSTM' or name == 'GRU' or name == 'CNN_LSTM' or name =="EN_1" or name =="EN_2" or name =="EN_3":
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        else:    
            X_test = rf_bk
        
        file = 'images/' + name + '.png'
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)


if __name__ == '__main__':
    main(sys.argv)
