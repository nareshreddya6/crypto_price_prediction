import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Activation, Dropout
# from keras.api import *
# from keras.api.layers import LSTM
import keras
from keras import Sequential
from keras.src.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

import tensorflow as tf

print(tf.__version__)  # Check TensorFlow version
print(tf.keras.__version__)  # Check Keras version within TensorFlow

def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    prices = data['close'].values.reshape(-1, 1)

    # Normalize prices to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    return scaled_prices, scaler




def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# Example usage:


def variables():
    scaled_prices, scaler = preprocess_data('data/historical_data.csv')
    time_step = 60  # Number of previous time steps to consider for prediction
    X, y = create_dataset(scaled_prices, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input

    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=32)
    return model,scaler,X
