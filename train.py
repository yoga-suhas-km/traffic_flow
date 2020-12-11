"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from process_data import load_data
import model
from keras.models import Model
from keras.callbacks import EarlyStopping
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
import _pickle as cPickle
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config, data):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    if name in ['lstm', 'gru', 'saes', 'cnn_lstm', 'en_1', 'en_2', 'en_3']:
        #model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        model.compile(loss="mse", optimizer="adam", metrics=['mse'])
        es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        if data == "pems":
            mc = ModelCheckpoint('model_pems/' + name + '.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
        elif data == "nyc":
            mc = ModelCheckpoint('model_nyc/' + name + '.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)


        hist = model.fit(
            X_train, y_train,
            batch_size=config["batch"],
            epochs=config["epochs"],
            validation_split=0.05,
            callbacks=[es, mc])

        #model.save('model/' + name + '.h5')
        df = pd.DataFrame.from_dict(hist.history)
        if data == "pems":
            df.to_csv('model_pems/' + name + ' loss.csv', encoding='utf-8', index=False)            
        elif data == "nyc":    
            df.to_csv('model_nyc/' + name + ' loss.csv', encoding='utf-8', index=False)            
           
    elif name == 'rf':
        model.fit(X_train, y_train)
        
        if data == "pems":
            with open('model_pems/' + name + '.h5', 'wb') as f:
                cPickle.dump(model, f)
        elif data == "nyc": 
            with open('model_nyc/' + name + '.h5', 'wb') as f:
                cPickle.dump(model, f)            

def train_seas(models, X_train, y_train, name, config, data):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            inputs = p.input
            outputs = p.get_layer('hidden').output
            hidden_layer_model = Model(input=inputs,output=outputs)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="adam", metrics=['mse'])

        print(temp.shape)
    
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              callbacks=[early])

        models[i] = m

    saes = models[-1]   

    for i in range(len(models) - 1):
        print(i)
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config, data)

def train_en(models, X_train, y_train, name, config, data):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    for i in range(len(models)):

        if i > 0:
            p = models[i]
            inputs = p.input
            outputs = p.get_layer('hidden%d' % (i)).output
            hidden_layer_model = Model(input=inputs,output=outputs)
        
        m = models[i]
        m.compile(loss="mse", optimizer="adam", metrics=['mse'])
    
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05,
              callbacks=[early])

        models[i] = m


    saes = models[-1]   

    train_model(saes, X_train, y_train, name, config, data)

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    parser.add_argument(
        "--data",
        default="pems",
        help="data to use.")        
    args = parser.parse_args()


    config = {"batch": 256, "epochs": 1000}

    if args.data == "pems":
        print("pems")
        X_train, X_test, y_train, y_test, scaler = load_data(data = "PEMS traffic prediction", force_download = False)
    elif args.data == "nyc":
        print("nyc")
        X_train, X_test, y_train, y_test, scaler = load_data(data = "nyc_bike_dataset", force_download = False)


    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([X_train.shape[1], 32, 32, 1])
        train_model(m, X_train, y_train, args.model, config, args.data)
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([X_train.shape[1], 32, 32, 1])
        train_model(m, X_train, y_train, args.model, config, args.data)
    if args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([X_train.shape[1], 32, 32, 32, 1])
        train_seas(m, X_train, y_train, args.model, config, args.data)
    if args.model == 'cnn_lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_cnn_lstm([X_train.shape[1], 32, 1])
        train_model(m, X_train, y_train, args.model, config, args.data)  
    if args.model == 'rf':
        m = model.get_rf()
        train_model(m, X_train, y_train, args.model, config, args.data)  
    if args.model == 'en_1':  
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_ensemble_en_1([X_train.shape[1], 32, 32, 1])
        train_en(m, X_train, y_train, args.model, config, args.data)
    if args.model == 'en_2':  
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_ensemble_en_2([X_train.shape[1], 32, 32, 1])
        train_en(m, X_train, y_train, args.model, config, args.data)
    if args.model == 'en_3':  
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_ensemble_en_2([X_train.shape[1], 32, 32, 1])
        train_en(m, X_train, y_train, args.model, config, args.data)      
        
if __name__ == '__main__':
    main(sys.argv)
