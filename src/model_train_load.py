# TensorFlow and Keras
from tensorflow import keras
from keras.regularizers import l2
from keras.utils import custom_object_scope
from keras.models import Model, load_model
from keras.layers import Dense, SimpleRNN, GRU, LSTM, Conv1D, Input, Concatenate, Flatten, Dropout, Add, AveragePooling1D

# Keras Temporal Convolutional Network (https://github.com/philipperemy/keras-tcn)
from tcn import TCN

# general purpose utilities
import pickle

# custom Python modules
from src import metrics as mt


def model_init(id_model, window_size, n_inputs, n_outputs):

  match id_model:

    case '1':
      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # SRNN layer
      srnn = SimpleRNN(64)(inputs)

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(srnn)

    case '2':
      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # GRU layer
      gru = GRU(64)(inputs)

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(gru)

    case '3':
      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # LSTM layer
      lstm = LSTM(64)(inputs)

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(lstm)

    case '4':
      # Input layer
      inputs = Input(shape=(window_size, n_inputs))

      # CONV1D layer
      conv1d = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)

      # Average Pooling layer
      avg_pool = AveragePooling1D(pool_size=2)(conv1d)

      # Flatten layer
      flatten = Flatten()(avg_pool)

      # Output layer
      outputs = Dense(n_outputs, activation='tanh')(flatten)

    case '5':
      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # 1st parallel branch (GRU)
      gru = GRU(64)(inputs)

      # 2nd parallel branch (LSTM)
      lstm = LSTM(64)(inputs)

      # 3rd parallel branch (CONV1D layer with Average Pooling and Flatten)
      conv1d = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
      avg_pool = AveragePooling1D(pool_size=2)(conv1d)
      flatten = Flatten()(avg_pool)

      # parallel layers concatenation
      output_concat = Concatenate()([gru, lstm, flatten])

      # Fully connected layers with skip connection
      dense_1 = Dense(32, activation='tanh')(output_concat)
      dense_2 = Dense(32, activation='tanh')(dense_1)
      skip_connection = Add()([dense_2, dense_1])

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(skip_connection)

    case '6':
      # define the NN regularization hyperparameters
      dropout_rate = 0.2
      l2_kernel_regulation = 0.02
      l2_bias_regulation = 0.02

      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # 1st parallel branch (GRU with dropout)
      gru = GRU(64, dropout=dropout_rate)(inputs)

      # 2nd parallel branch (LSTM with dropout)
      lstm = LSTM(64, dropout=dropout_rate)(inputs)

      # 3rd parallel branch (CONV1D layer with L2 regularization, Average Pooling and Flatten)
      conv1d = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_kernel_regulation), bias_regularizer=l2(l2_bias_regulation))(inputs)
      avg_pool = AveragePooling1D(pool_size=2)(conv1d)
      flatten = Flatten()(avg_pool)

      # parallel layers concatenation
      output_concat = Concatenate()([gru, lstm, flatten])

      # Fully connected layers with skip connection
      dense_1 = Dense(32, activation='tanh')(output_concat)
      dense_2 = Dense(32, activation='tanh')(dense_1)
      skip_connection = Add()([dense_2, dense_1])

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(skip_connection)

    case '7':
      # define the NN regularization hyperparameters
      dropout_rate = 0.2

      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # 1st LSTM layer (with Dropout)
      lstm_1 = LSTM(16, activation='tanh', return_sequences=True, dropout=dropout_rate)(inputs)

      # 2nd LSTM layer (with Dropout)
      lstm_2 = LSTM(32, activation='tanh', return_sequences=True, dropout=dropout_rate)(lstm_1)

      # 3rd LSTM layer
      lstm_3 = LSTM(64, activation='tanh', return_sequences=True)(lstm_2)

      # 4th LSTM layer
      lstm_4 = LSTM(128, activation='tanh')(lstm_3)

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(lstm_4)
    
    case '8':
      # define the NN regularization hyperparameters
      dropout_rate = 0.2

      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # 1st LSTM layer (with Dropout)
      lstm_1 = LSTM(128, activation='tanh', return_sequences=True, dropout=dropout_rate)(inputs)

      # 2nd LSTM layer (with Dropout)
      lstm_2 = LSTM(64, activation='tanh', return_sequences=True, dropout=dropout_rate)(lstm_1)

      # 3rd LSTM layer
      lstm_3 = LSTM(32, activation='tanh', return_sequences=True)(lstm_2)

      # 4th LSTM layer
      lstm_4 = LSTM(16, activation='tanh')(lstm_3)

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(lstm_4)
    
    case '9':
      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # TCN (Temporal Convolutional Network) layer
      tcn = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32], activation='relu', return_sequences=True)(inputs)

      # Flatten layer
      flatten = Flatten()(tcn)
      
      # output layer
      outputs = Dense(n_outputs, activation='tanh')(flatten)
    
    case '10a':
      # define the NN regularization hyperparameters
      l2_kernel_regulation = 0.02
      l2_bias_regulation = 0.02

      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # 1st parallel TCN (Temporal Convolutional Network)
      tcn_1 = TCN(nb_filters=16, kernel_size=6, dilations=[1, 2, 4, 8], activation='relu', return_sequences=True)(inputs)
      flatten_1 = Flatten()(tcn_1)

      # 2nd parallel TCN (Temporal Convolutional Network)
      tcn_2 = TCN(nb_filters=16, kernel_size=6, dilations=[1, 2, 4, 8, 16], activation='relu', return_sequences=True)(inputs)
      flatten_2 = Flatten()(tcn_2)

      # 3rd parallel TCN (Temporal Convolutional Network)
      tcn_3 = TCN(nb_filters=32, kernel_size=6, dilations=[1, 2, 4, 8], activation='relu', return_sequences=True)(inputs)
      flatten_3 = Flatten()(tcn_3)

      # 4th parallel TCN (Temporal Convolutional Network)
      tcn_4 = TCN(nb_filters=32, kernel_size=6, dilations=[1, 2, 4, 8, 16], activation='relu', return_sequences=True)(inputs)
      flatten_4 = Flatten()(tcn_4)

      # parallel layers concatenation
      output_concat = Concatenate()([flatten_1, flatten_2, flatten_3, flatten_4])

      # Fully connected layers with skip connection
      dense_1 = Dense(32, activation='tanh', kernel_regularizer=l2(l2_kernel_regulation), bias_regularizer=l2(l2_bias_regulation))(output_concat)
      dense_2 = Dense(32, activation='tanh')(dense_1)
      skip_connection = Add()([dense_2, dense_1])

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(skip_connection)
    
    case '10b':
      # define the NN regularization hyperparameters
      dropout_rate = 0.1

      # input layer
      inputs = Input(shape=(window_size, n_inputs))

      # 1st parallel TCN (Temporal Convolutional Network)
      tcn_1 = TCN(nb_filters=16, kernel_size=6, dilations=[1, 2, 4, 8], activation='relu', dropout_rate=dropout_rate, return_sequences=True)(inputs)
      flatten_1 = Flatten()(tcn_1)

      # 2nd parallel TCN (Temporal Convolutional Network)
      tcn_2 = TCN(nb_filters=16, kernel_size=6, dilations=[1, 2, 4, 8, 16], activation='relu', dropout_rate=dropout_rate, return_sequences=True)(inputs)
      flatten_2 = Flatten()(tcn_2)

      # 3rd parallel TCN (Temporal Convolutional Network)
      tcn_3 = TCN(nb_filters=32, kernel_size=6, dilations=[1, 2, 4, 8], activation='relu', dropout_rate=dropout_rate, return_sequences=True)(inputs)
      flatten_3 = Flatten()(tcn_3)

      # 4th parallel TCN (Temporal Convolutional Network)
      tcn_4 = TCN(nb_filters=32, kernel_size=6, dilations=[1, 2, 4, 8, 16], activation='relu', dropout_rate=dropout_rate, return_sequences=True)(inputs)
      flatten_4 = Flatten()(tcn_4)

      # parallel layers concatenation
      output_concat = Concatenate()([flatten_1, flatten_2, flatten_3, flatten_4])

      # Fully connected layers with skip connection
      dense_1 = Dense(32, activation='tanh')(output_concat)
      dense_2 = Dense(32, activation='tanh')(dense_1)
      skip_connection = Add()([dense_2, dense_1])

      # output layer
      outputs = Dense(n_outputs, activation='tanh')(skip_connection)
   
    case _:
      # notify the error and quit the function
      print("The selected model is not valid/available.")
      return



  # build the model to return
  model = Model(inputs=inputs, outputs=outputs)

  return model


def model_build(id_model, window_size, lr, loss, metrics, n_inputs=2, n_outputs=2):

  model = model_init(id_model, window_size, n_inputs, n_outputs)

  optimizer = keras.optimizers.Adam(learning_rate=lr)                   
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  return model


def load_trained_model(model_path, history_path):

    model = load_model(model_path)

    history = pickle.load(open(history_path, 'rb'))

    return model, history


def model_train(model, training_generator, validation_generator, n_epochs, callbacks, model_checkpoints_path):

  # Addestramento della rete neurale
  history = model.fit(training_generator, epochs=n_epochs, callbacks=callbacks, verbose=1, validation_data=validation_generator)

  model.save(model_checkpoints_path + '/last_model.h5')
  pickle.dump(history, open(model_checkpoints_path + '/train_history.pkl', 'wb'))

  return history
  

def lr_search(model, training_generator, n_epochs, validation_generator, lr_base_coeff, lr_exp_coeff, model_checkpoints_path):

  lr_schedule = keras.callbacks.LearningRateScheduler(
      lambda epoch: lr_base_coeff * 10**(epoch * lr_exp_coeff)
  )

  callbacks=[lr_schedule]

  history = model_train(model, training_generator, validation_generator, n_epochs, callbacks, model_checkpoints_path)

  return history
