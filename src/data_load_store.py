# data analysis and manipulation
import pandas as pd

# tensor operations
import numpy as np

# data I/O utilities
import os


def load_dataset(dataset_path):

  dir = os.listdir(dataset_path)
  dir.sort()

  df = []               # dataframe where to load the dataset
  samples_count = 0     # counter of dataset samples
  for file in dir:

    path = dataset_path + '/' + file
    df.append(pd.read_csv(path, header=None))
    samples_count += 1

  df = pd.concat(df, axis=1)
  np_dataset = pd.DataFrame.to_numpy(df)

  return np_dataset.T, samples_count


def store_dataset(dataset_path, dataset, samples_len):

  dir = os.listdir(dataset_path)
  dir.sort()

  n_digits_filename = 3
  n_samples = int(dataset.shape[0] / samples_len)

  for i in range(n_samples):

    idx_lb = n_samples * i
    idx_ub = n_samples * (i + 1)

    data = dataset[idx_lb:idx_ub, :]
    filename = str(i).zfill(n_digits_filename)
    data_path = dataset_path + '/' + filename + '.csv'

    np.savetxt(data_path, data, delimiter=',')

