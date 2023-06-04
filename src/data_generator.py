# TensorFlow and Keras
from tensorflow import keras
from keras.utils import Sequence

# tensor operations
import numpy as np

class SysIdentDataGenerator(Sequence):
    """Data generator that, given a dataset of raw input and ouput measurements of a physical system, parses them into a suitable format to
    train a system identification model and make predictions with it.
    """

    def __init__(self, X_set_path, Y_set_path, n_samples, samples_len, window_size=10, batch_size=32, to_fit=True, shuffle=True, load_dataset_first=True):
        """Data generator initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """ 
        self.X_set_path = X_set_path
        self.Y_set_path = Y_set_path
        self.n_samples = n_samples
        self.samples_len = samples_len
        self.window_size = window_size
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.load_dataset_first = load_dataset_first

        self.windows_per_sample = samples_len - window_size
        self.n_windows = n_samples * self.windows_per_sample
        self.indexes = np.arange(self.n_windows)

        if (self.load_dataset_first):
          self.X_set = np.loadtxt(self.X_set_path, delimiter=',')
          self.Y_set = np.loadtxt(self.Y_set_path, delimiter=',')

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
        
        X = []

        for id in batch_indexes:

          dt = int(id / self.windows_per_sample) * self.samples_len + (id % self.windows_per_sample)

          if (self.load_dataset_first):
            X_set = self.X_set[dt:dt+self.window_size, :]
          else:
            X_set = np.loadtxt(self.X_set_path, skiprows=dt, max_rows=self.window_size, delimiter=',')

          features = [elem for elem in X_set]
          X.append(features)

        return np.array(X)
    
    def _generate_Y(self, batch_indexes):
        
        Y = []

        for id in batch_indexes:

          dt = int(id / self.windows_per_sample) * self.samples_len + (id % self.windows_per_sample)

          if (self.load_dataset_first):
            Y_set = self.Y_set[dt+self.window_size, :]
          else:
            Y_set = np.loadtxt(self.Y_set_path, skiprows=dt+self.window_size, max_rows=1, delimiter=',')

          labels = Y_set
          Y.append(labels)

        return np.array(Y)


    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

