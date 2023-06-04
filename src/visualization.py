# Keras
from tensorflow import keras

# data analysis and manipulation
import matplotlib.pyplot as plt

# tensor operations
import numpy as np

# general purpose utilities
from IPython.display import Image, display


def show_dataset_samples(id_train,          id_valid,           id_test,
                         train_set,         valid_set,          test_set,
                         train_samples_len, valid_samples_len,  test_samples_len):

  """Visualize examples of samples from the dataset.

  @type id_train: int
  @param id_train: Index of the training sample to show.
  @type id_valid: int
  @param id_valid: Index of the validation sample to show.
  @type id_test: int
  @param id_test: Index of the testing sample to show.
  @type train_set: numpy.ndarray
  @param train_set: Training dataset.
  @type valid_set: numpy.ndarray
  @param valid_set: Validation dataset.
  @type test_set: numpy.ndarray
  @param test_set: Testing dataset.
  @type train_samples_len: int
  @param train_samples_len: Length of each training sample.
  @type valid_samples_len: int
  @param valid_samples_len: Length of each validation sample.
  @type test_samples_len: int
  @param test_samples_len: Length of each testing sample.
  """
  
  # determine the range of indices for the samples to show (for train, validation and test sets)
  train_sample_to_show  = np.arange(id_train * train_samples_len,  (id_train + 1) * train_samples_len)
  valid_sample_to_show  = np.arange(id_valid * valid_samples_len,  (id_valid + 1) * valid_samples_len)
  test_sample_to_show   = np.arange(id_test * test_samples_len,    (id_test + 1) * test_samples_len)

  # dataset metadata
  n_inputs  = 2     # input measurements:   time [seconds], analog control signal [voltage]
  n_outputs = 2     # output measurements:  yaw position [degrees], yaw velocity [degrees/seconds]
  n_sets    = 3     # datasets:             training, validation, testing

  # configure the plot
  rows = n_inputs + n_outputs
  cols = n_sets
  fig_width   = 16
  fig_height  = 10

  # create subplots for each input and output measurement
  fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))

  # plot the samples for each input and output measurement
  for i in range(rows):
    axes[i,0].plot(train_set[train_sample_to_show,i])
    axes[i,1].plot(valid_set[valid_sample_to_show,i])
    axes[i,2].plot(test_set[test_sample_to_show,i])

  # set y-axis labels for each measurement type
  axes[0,0].set_ylabel('Time [s]')
  axes[1,0].set_ylabel('Volt [V]')
  axes[2,0].set_ylabel('Yaw position [deg]')
  axes[3,0].set_ylabel('Yaw velocity [deg/s]')

  # set titles for each dataset type
  axes[0,0].set_title('Train sample')
  axes[0,1].set_title('Validation sample')
  axes[0,2].set_title('Test sample')

  # add a grid to each subplot
  for i in range(rows):
    for j in range(cols):
      axes[i,j].grid()

  # show the plot
  plt.show()


def plot_prediction(id_pred, Y_pred, Y_true, t_span, samples_len, window_size, dataset_name):

  """Plot the model's predictions and the ground truth for a specific prediction sample.

  @type id_pred: int
  @param id_pred: Index of the prediction sample to plot.
  @type Y_pred: numpy.ndarray
  @param Y_pred: Predicted outputs.
  @type Y_true: numpy.ndarray
  @param Y_true: Ground truth outputs.
  @type t_span: numpy.ndarray
  @param t_span: Time span of the dataset.
  @type samples_len: int
  @param samples_len: Length of each sample in the dataset.
  @type window_size: int
  @param window_size: Size of the window used for prediction.
  @type dataset_name: str
  @param dataset_name: Name of the dataset.
  """

  # configure the plot
  rows = 2
  cols = 1
  fig_width   = 16
  fig_height  = 8

  # create subplots for each output measurement
  fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))

  # get the time arrays of both the prediction and the ground truth
  t_true = t_span[samples_len*(id_pred) : samples_len*(id_pred + 1)]
  t_pred = t_true[window_size:]

  # plot the predictions and ground truth for each output measurement
  for i in range(rows):
    axes[i].plot(t_pred, Y_pred[(samples_len - window_size)*(id_pred) : (samples_len - window_size)*(id_pred + 1), i])
    axes[i].plot(t_true, Y_true[samples_len*(id_pred) : samples_len*(id_pred + 1), i])

  # add legends and labels to the subplots
  axes[0].legend(['Yaw pos prediction', 'Yaw pos ground truth'], loc='best')
  axes[1].legend(['Yaw vel prediction', 'Yaw vel ground truth'], loc='best')

  axes[1].set_xlabel('Time [s]')
  axes[0].set_ylabel('Yaw position [deg]')
  axes[1].set_ylabel('Yaw velocity [deg/s]')

  # add a grid to the subplots
  for i in range(rows):
    axes[i].grid()

  # set the title of the plot
  axes[0].set_title('Model multi-step prediction on \'' + dataset_name + '\' dataset')

  # show the plot
  plt.show()


def plot_train_metrics(history):

  """Plot the training metrics (loss, MAE, R^2) during training.

  @type history: keras.callbacks.History
  @param history: History object containing the training metrics.
  """
  
  # configure the plot
  rows = 1
  cols = 3
  fig_width   = 16
  fig_height  = 4

  # create subplots for each metric
  fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))

  # plot loss (MSE)
  axes[0].plot(history.history['loss'])
  axes[0].plot(history.history['val_loss'])
  axes[0].set_title('Loss: Mean Squared Error (MSE)')
  axes[0].legend(['Training', 'Validation'], loc='best')
  axes[0].set_ylabel('Metric')
  axes[0].set_xlabel('Epoch')
  axes[0].grid()

  # plot MAE
  axes[1].plot(history.history['mae'])
  axes[1].plot(history.history['val_mae'])
  axes[1].set_title('Mean Absolute Error (MAE)')
  axes[1].legend(['Training', 'Validation'], loc='best')
  axes[1].set_xlabel('Epoch')
  axes[1].grid()

  # plot R^2
  axes[2].plot(history.history['r_squared'])
  axes[2].plot(history.history['val_r_squared'])
  axes[2].set_title('Coefficient of determination (R\u00B2)')
  axes[2].legend(['Training', 'Validation'], loc='best')
  axes[2].set_xlabel('Epoch')
  axes[2].grid()

  # show the plot
  plt.show()
  

def show_model(model, model_checkpoints_path):
  
  """Show the summary and architecture plot of a trained/loaded model.

  @type model: keras.models.Model
  @param model: Trained model to show.
  @type model_checkpoints_path: str
  @param model_checkpoints_path: Path to the directory where the model checkpoints are saved.
  """
  
  # print model summary
  print("************************************")
  print("*                                  *")
  print("*    MODEL ARCHITECTURE SUMMARY    *")
  print("*                                  *")
  print("************************************\n")
  model.summary()
  print("\n")

  # show model architecture plot
  print("*********************************")
  print("*                               *")
  print("*    MODEL ARCHITECTURE PLOT    *")
  print("*                               *")
  print("*********************************\n")
  keras.utils.plot_model(model, model_checkpoints_path + '/architecture.png', show_shapes=True)
  display(Image(filename = model_checkpoints_path + '/architecture.png'))


def show_lr_search(history, n_epochs, lr_base_coeff, lr_exp_coeff, plot_padding_perc=50):

  """Show the Learning Rate (LR) search plot.

  @type history: keras.callbacks.History
  @param history: History object containing the training metrics during the learning rate search.
  @type n_epochs: int
  @param n_epochs: Number of epochs during the learning rate search.
  @type lr_base_coeff: float
  @param lr_base_coeff: Base coefficient for the learning rate.
  @type lr_exp_coeff: float
  @param lr_exp_coeff: Exponential coefficient for the learning rate.
  @type plot_padding_perc: int
  @param plot_padding_perc: Padding percentage for the plot axis limits.
  """
  
  # get the list of all scheduled LRs
  lrs = lr_base_coeff * (10**(np.arange(n_epochs) * lr_exp_coeff))

  # get lower and upper bounds of the LR-loss plot
  plt_lr_lb = lrs[0]  * (1 - plot_padding_perc/100)
  plt_lr_ub = lrs[-1] * (1 + plot_padding_perc/100)
  plt_loss_lb = 0
  plt_loss_ub = history.history['loss'][0] * (1 + plot_padding_perc/100)
  
  # set the desired width and height of the plot
  fig_width   = 8
  fig_height  = 3
  fig = plt.figure(figsize=(fig_width, fig_height))

  # show plot, title, labels, legend and grid
  plt.semilogx(lrs, history.history['loss'])
  plt.title('Loss (MSE) with an increasing Learning Rate (LR)')
  plt.xlabel('Learning Rate (LR)')
  plt.ylabel('Loss (MSE)')
  plt.legend(['Training loss'], loc='best')
  plt.grid()
  plt.axis([plt_lr_lb, plt_lr_ub, plt_loss_lb, plt_loss_ub])
  

def show_mse_pred(model_purpose, mse):

  """Show the Mean Squared Error (MSE) of the model's predictions.

  @type model_purpose: str
  @param model_purpose: Purpose of the model ('control' or 'prediction').
  @type mse: list[float]
  @param mse: List of MSE values for different data scenarios.
  """
  
  if (model_purpose == 'control'):
    print("MSE of yaw position on \'delta\' data:       " + str(mse[0]))
    print("MSE of yaw velocity on \'delta\' data:       " + str(mse[1]))
    print("MSE of yaw position on \'contiguous\' data:  " + str(mse[2]))
    print("MSE of yaw velocity on \'contiguous\' data:  " + str(mse[3]))
  elif (model_purpose == 'prediction'):
    print("MSE of yaw position on \'contiguous\' data:  " + str(mse[0]))
    print("MSE of yaw velocity on \'contiguous\' data:  " + str(mse[1]))
  else:
    print("Error: model purpose is not valid (it should be \'control\' or \'prediction\')")

