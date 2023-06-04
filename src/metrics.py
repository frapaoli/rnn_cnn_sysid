# TensorFlow and Keras
from tensorflow import keras
from keras import backend as K

# tensor operations
import numpy as np


def r_squared(y_true, y_pred):

    """Calculate the coefficient of determination (R-squared) between the true values and predicted values.

    @type y_true: tensor or variable
    @param y_true: True values.
    @type y_pred: tensor or variable
    @param y_pred: Predicted values.
    @rtype: tensor or variable
    @returns: R-squared value.
    """

    # residual sum of squares
    SS_res = K.sum(K.square(y_true - y_pred))

    # total sum of squares
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return (1 - SS_res/(SS_tot + K.epsilon()))


def mse_pred(Y_pred, Y_true, samples_len, window_size):

  """Calculate the Mean Squared Error (MSE) between the predicted values and true values.

  @type Y_pred: numpy.ndarray
  @param Y_pred: Predicted values.
  @type Y_true: numpy.ndarray
  @param Y_true: True values.
  @type samples_len: int
  @param samples_len: Length of each sample.
  @type window_size: int
  @param window_size: Size of the window used for prediction.
  @rtype: numpy.ndarray
  @returns: MSE values for each target variable.
  """
  
  # number of predicted values (for each target variable)
  n_preds = int(Y_true.shape[0] / samples_len)

  # numpy array that will contain all MSE (one for each target variable)
  mse = np.zeros((1, Y_true.shape[1]))

  # iterate over the predictions made by the model
  for i in range(n_preds):

    # compute the single prediction and its related ground truth
    y_pred = Y_pred[(samples_len - window_size)*(i) : (samples_len - window_size)*(i + 1), :]
    y_true = Y_true[samples_len*(i) : samples_len*(i + 1), :]
    y_true = y_true[window_size:, :]

    # compute the sum of all squared errors
    error = np.subtract(y_true, y_pred)
    error_squared = np.square(error)
    sum = np.sum(error_squared, axis=0)
    mse += np.sqrt(sum)

  # compute the MSE
  mse /= n_preds
  
  return mse

