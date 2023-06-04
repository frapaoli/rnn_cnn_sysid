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

  train_sample_to_show  = np.arange(id_train * train_samples_len,  (id_train + 1) * train_samples_len)
  valid_sample_to_show  = np.arange(id_valid * valid_samples_len,  (id_valid + 1) * valid_samples_len)
  test_sample_to_show   = np.arange(id_test * test_samples_len,    (id_test + 1) * test_samples_len)

  # dataset metadata
  n_inputs  = 2     # input measurements:   time [seconds], analog control signal [voltage]
  n_outputs = 2     # output measurements:  yaw position [degrees], yaw velocity [degrees/seconds]
  n_sets    = 3     # datasets:             training, validation, testing

  rows = n_inputs + n_outputs
  cols = n_sets
  fig_width   = 16
  fig_height  = 10

  fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))

  for i in range(rows):
    axes[i,0].plot(train_set[train_sample_to_show,i])
    axes[i,1].plot(valid_set[valid_sample_to_show,i])
    axes[i,2].plot(test_set[test_sample_to_show,i])

  axes[0,0].set_ylabel('Time [s]')
  axes[1,0].set_ylabel('Volt [V]')
  axes[2,0].set_ylabel('Yaw position [deg]')
  axes[3,0].set_ylabel('Yaw velocity [deg/s]')

  axes[0,0].set_title('Train sample')
  axes[0,1].set_title('Validation sample')
  axes[0,2].set_title('Test sample')

  for i in range(rows):
    for j in range(cols):
      axes[i,j].grid()

  plt.show()


def plot_prediction(id_pred, Y_pred, Y_true, t_span, samples_len, window_size, dataset_name):

  rows = 2      # yaw and yawd
  cols = 1
  fig_width   = 16
  fig_height  = 8

  fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))

  t_true = t_span[samples_len*(id_pred) : samples_len*(id_pred + 1)]
  t_pred = t_true[window_size:]

  for i in range(rows):
    axes[i].plot(t_pred, Y_pred[(samples_len - window_size)*(id_pred) : (samples_len - window_size)*(id_pred + 1), i])
    axes[i].plot(t_true, Y_true[samples_len*(id_pred) : samples_len*(id_pred + 1), i])

  axes[0].legend(['Yaw pos prediction', 'Yaw pos ground truth'], loc='best')
  axes[1].legend(['Yaw vel prediction', 'Yaw vel ground truth'], loc='best')

  axes[1].set_xlabel('Time [s]')
  axes[0].set_ylabel('Yaw position [deg]')
  axes[1].set_ylabel('Yaw velocity [deg/s]')

  for i in range(rows):
    axes[i].grid()

  axes[0].set_title('Model multi-step prediction on \'' + dataset_name + '\' dataset')

  plt.show()


def plot_train_metrics(history):

  rows = 1
  cols = 3
  fig_width   = 16
  fig_height  = 4

  fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))

  axes[0].plot(history.history['loss'])
  axes[0].plot(history.history['val_loss'])
  axes[0].set_title('Loss: Mean Squared Error (MSE)')
  axes[0].legend(['Training', 'Validation'], loc='best')
  axes[0].set_ylabel('Metric')
  axes[0].set_xlabel('Epoch')
  axes[0].grid()

  axes[1].plot(history.history['mae'])
  axes[1].plot(history.history['val_mae'])
  axes[1].set_title('Mean Absolute Error (MAE)')
  axes[1].legend(['Training', 'Validation'], loc='best')
  axes[1].set_xlabel('Epoch')
  axes[1].grid()

  axes[2].plot(history.history['r_squared'])
  axes[2].plot(history.history['val_r_squared'])
  axes[2].set_title('Coefficient of determination (R\u00B2)')
  axes[2].legend(['Training', 'Validation'], loc='best')
  axes[2].set_xlabel('Epoch')
  axes[2].grid()

  plt.show()
  

def show_model(model, model_checkpoints_path):
  
  print("************************************")
  print("*                                  *")
  print("*    MODEL ARCHITECTURE SUMMARY    *")
  print("*                                  *")
  print("************************************\n")
  model.summary()
  print("\n")

  print("*********************************")
  print("*                               *")
  print("*    MODEL ARCHITECTURE PLOT    *")
  print("*                               *")
  print("*********************************\n")
  keras.utils.plot_model(model, model_checkpoints_path + '/architecture.png', show_shapes=True)
  display(Image(filename = model_checkpoints_path + '/architecture.png'))


def show_lr_search(history, n_epochs, lr_base_coeff, lr_exp_coeff, plot_padding_perc=50):

  lrs = lr_base_coeff * (10**(np.arange(n_epochs) * lr_exp_coeff))

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




