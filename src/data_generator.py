# TensorFlow and Keras
from tensorflow import keras
from keras.utils import Sequence

# tensor operations
import numpy as np


class SysIdentDataGenerator(Sequence):
    
    """Data generator that, given a dataset of raw input and ouput measurements of a physical system, parses them into a suitable format
    (i.e., time-windows) to train a system identification model and make predictions with it.
    """

    def __init__(self, X_set_path, Y_set_path, n_samples, samples_len, window_size=10, batch_size=32, to_fit=True, shuffle=True, load_dataset_first=True):
        
        """Initializes the SysIdentDataGenerator object.

        @type X_set_path: str
        @param X_set_path: Path to the input measurements dataset.
        @type Y_set_path: str
        @param Y_set_path: Path to the output measurements dataset.
        @type n_samples: int
        @param n_samples: Number of samples (i.e., data acquisitions) in the dataset.
        @type samples_len: int
        @param samples_len: Length (i.e., number of elements) of each sample in the dataset.
        @type window_size: int
        @param window_size: Size of the time window used for training and prediction.
        @type batch_size: int
        @param batch_size: Size of the batches to generate.
        @type to_fit: bool
        @param to_fit: Flag indicating whether the generator is used for fitting or prediction.
        @type shuffle: bool
        @param shuffle: Flag indicating whether to shuffle the indexes of the dataset at the end of each epoch.
        @type load_dataset_first: bool
        @param load_dataset_first: Flag indicating whether to load the dataset into memory at initialization.
        """

        self.X_set_path = X_set_path      # path to the input measurements dataset
        self.Y_set_path = Y_set_path      # path to the output measurements dataset
        self.n_samples = n_samples        # number of samples in the dataset
        self.samples_len = samples_len    # length of each sample in the dataset
        self.window_size = window_size    # size of the time window
        self.to_fit = to_fit              # flag indicating whether the generator is used for fitting or prediction
        self.batch_size = batch_size      # size of the batches to generate
        self.shuffle = shuffle            # flag indicating whether to shuffle the indexes of the dataset at the end of each epoch
        self.load_dataset_first = load_dataset_first    # flag indicating whether to load the dataset into memory at initialization

        self.windows_per_sample = samples_len - window_size   # number of windows per sample
        self.n_windows = n_samples * self.windows_per_sample  # total number of windows in the dataset
        self.indexes = np.arange(self.n_windows)              # array of indexes for the windows

        if (self.load_dataset_first):
          # load the dataset into memory
          self.X_set = np.loadtxt(self.X_set_path, delimiter=',')
          self.Y_set = np.loadtxt(self.Y_set_path, delimiter=',')

        # shuffle the dataset indexes
        self.on_epoch_end()

    def __len__(self):
        
        """Returns the number of batches per epoch

        @rtype: int
        @returns: number of batches per epoch
        """

        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        
        """Generates and returns one batch of data

        @type index: int
        @param index: index of the batch
        @rtype: numpy.ndarray
        @returns: training data X and Y (containing batch_size data) if the data generator is being used to fit the model, otherwise only X (in order to make a prediction).
        """

        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size : (index + 1)*self.batch_size]

        # Generate data
        X = self._generate_X(batch_indexes)

        if self.to_fit:
            Y = self._generate_Y(batch_indexes)
            return X, Y
        
        return X

    def _generate_X(self, batch_indexes):
        
        """Generate the input data (X) for the given batch indexes.

        @type batch_indexes: numpy.ndarray
        @param batch_indexes: Array of indexes representing the windows in the batch.
        @rtype: numpy.ndarray
        @returns: Array of input data (X) for the batch.
        """
        
        # list to store the input data for each window in the batch
        X = []

        # iterate over the selected batches
        for id in batch_indexes:

          # compute the time instants relative to the currently indexed batch
          dt = int(id / self.windows_per_sample) * self.samples_len + (id % self.windows_per_sample)

          # get X_set relative to the currently indexed batch
          if (self.load_dataset_first):
            X_set = self.X_set[dt:dt+self.window_size, :]
          else:
            X_set = np.loadtxt(self.X_set_path, skiprows=dt, max_rows=self.window_size, delimiter=',')

          # extract the features from X_set
          features = [elem for elem in X_set]

          # append the features to the list
          X.append(features)

        # convert the list of features to a numpy array, and return it
        return np.array(X)
    
    def _generate_Y(self, batch_indexes):
        
        """Generate the output data (Y) for the given batch indexes.

        @type batch_indexes: numpy.ndarray
        @param batch_indexes: Array of indexes representing the windows in the batch.
        @rtype: numpy.ndarray
        @returns: Array of output data (Y) for the batch.
        """
        
        # list to store the output data for each window in the batch
        Y = []

        # iterate over the selected batches
        for id in batch_indexes:
          
          # compute the time instants relative to the currently indexed batch
          dt = int(id / self.windows_per_sample) * self.samples_len + (id % self.windows_per_sample)

          # get Y_set relative to the currently indexed batch
          if (self.load_dataset_first):
            Y_set = self.Y_set[dt+self.window_size, :]
          else:
            Y_set = np.loadtxt(self.Y_set_path, skiprows=dt+self.window_size, max_rows=1, delimiter=',')

          # extract the labels from Y_set
          labels = Y_set

          # append the labels to the list
          Y.append(labels)

        # convert the list of labels to a numpy array, and return it
        return np.array(Y)

    def on_epoch_end(self):
        
        """Updates indexes after each epoch
        """

        if self.shuffle == True:
            # shuffle indexes after each epoch
            np.random.shuffle(self.indexes)

