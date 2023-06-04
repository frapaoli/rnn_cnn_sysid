# data analysis and manipulation
import pandas as pd

# tensor operations
import numpy as np

# data I/O utilities
import os


def load_dataset(dataset_path):

  """Load the dataset from the given directory path.

  @type dataset_path: str
  @param dataset_path: Path to the directory containing the dataset files.
  @rtype: tuple(numpy.ndarray, int)
  @returns: Tuple containing the loaded dataset as a numpy array and the number of samples in the dataset.
  """
  
  # get the list of files in the directory
  dir = os.listdir(dataset_path)
  dir.sort()

  df = []               # dataframe where to load the dataset
  samples_count = 0     # counter of dataset samples

  # iterate over all files in the directory
  for file in dir:

    # get the file's content
    path = dataset_path + '/' + file
    df.append(pd.read_csv(path, header=None))
    samples_count += 1

  # concatenate the dataframes along the columns
  df = pd.concat(df, axis=1)

  # convert the dataframe to a numpy array
  np_dataset = pd.DataFrame.to_numpy(df)

  # transpose the loaded dataset array, and return it with the sample count
  return np_dataset.T, samples_count


def store_dataset(dataset_path, dataset, samples_len):

  """Store the dataset in the given directory path.

  @type dataset_path: str
  @param dataset_path: Path to the directory where the dataset files will be stored.
  @type dataset: numpy.ndarray
  @param dataset: Dataset to be stored as a numpy array.
  @type samples_len: int
  @param samples_len: Length of each sample in the dataset.
  """
  
  # get the list of files in the directory
  dir = os.listdir(dataset_path)
  dir.sort()

  n_digits_filename = 3     # number of digits that we want in the file's name
  n_samples = int(dataset.shape[0] / samples_len)   # compute the number of dataset samples

  # iterate over each sample
  for i in range(n_samples):
    
    # get lower and upper bound indexes of the currently considered sample
    idx_lb = n_samples * i
    idx_ub = n_samples * (i + 1)

    data = dataset[idx_lb:idx_ub, :]    # get the sample
    filename = str(i).zfill(n_digits_filename)    # ensure that the file's name has a minimum of 3 digits
    data_path = dataset_path + '/' + filename + '.csv'    # get the path where to store the sample

    # save the sample as a CSV file
    np.savetxt(data_path, data, delimiter=',')

