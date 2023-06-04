# TensorFlow and Keras
from tensorflow import keras
from keras import backend as K

# tensor operations
import numpy as np


def r_squared(y_true, y_pred):

    # residual sum of squares
    SS_res = K.sum(K.square(y_true - y_pred))

    # total sum of squares
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return (1 - SS_res/(SS_tot + K.epsilon()))


def mse_pred(Y_pred, Y_true, samples_len, window_size):

  n_preds = int(Y_true.shape[0] / samples_len)

  mse = np.zeros((1, Y_true.shape[1]))

  for i in range(n_preds):

    y_pred = Y_pred[(samples_len - window_size)*(i) : (samples_len - window_size)*(i + 1), :]
    y_true = Y_true[samples_len*(i) : samples_len*(i + 1), :]
    y_true = y_true[window_size:, :]

    error = np.subtract(y_true, y_pred)
    error_squared = np.square(error)
    sum = np.sum(error_squared, axis=0)
    mse += np.sqrt(sum)

  mse /= n_preds
  
  return mse