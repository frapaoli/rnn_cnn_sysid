# data analysis and manipulation
from sklearn.preprocessing import MinMaxScaler

# tensor operations
import numpy as np


def contig_to_delta_meas(contig_meas, n_samples, samples_len):
  
  """Given a tensor of contiguous time-varying measurements, returns a tensor representing the 'delta' of those measurements
  (i.e., for each pair of subsequent values in the input tensor, the output tensor will store their subtraction).

  @type contig_meas: numpy.ndarray
  @param contig_meas: NumPy 2D tensor that stores in each column an array of contiguous time-varying measurements.
      Each column contains measurements of different physical meaning (e.g., Volt, degrees), and each column can contain
      measurements obtained through different data acquisition experiments.
  @type n_samples: int
  @param n_samples: number of samples (i.e., of separated data acquisitions) in each contig_meas column.
  @type samples_len: int
  @param samples_len: length (i.e., number of elements) of each sample in contig_meas columns.
  @rtype: numpy.ndarray
  @returns: NumPy 2D tensor with same dimensions as contig_meas, storing in each column the 'delta' version of it.
      In particular, separately for each column, the function takes all pairs of subsequent measurements (belonging to the same
      data acquisition experiment) and returns the variation between the two values.
  """

  # create an empty 2D tensor for the 'delta' data
  delta_meas = np.empty([contig_meas.shape[0], contig_meas.shape[1]])

  # generate the 'delta' data
  delta_meas[:,0]  = contig_meas[:,0]
  delta_meas[:,1:] = _contig_to_delta(contig_meas[:,1:], n_samples, samples_len)

  return delta_meas


def _contig_to_delta(contig, n_samples, samples_len):
  
  """Computes the 'delta' of a given tensor of contiguous time-varying measurements.

  @type contig: numpy.ndarray
  @param contig: NumPy 2D tensor that stores in each column an array of contiguous time-varying measurements.
  @type n_samples: int
  @param n_samples: number of samples (i.e., of separated data acquisitions) in each contig column.
  @type samples_len: int
  @param samples_len: length (i.e., number of elements) of each sample in contig columns.
  @rtype: numpy.ndarray
  @returns: NumPy 2D tensor with same dimensions as contig, representing the 'delta' of the input measurements.
  """

  # compute the difference between subsequent measurements along the first axis
  delta = np.diff(contig, axis=0)

  # insert a row of zeros at the beginning of the delta tensor to maintain the same shape
  delta = np.insert(delta, 0, np.zeros(delta.shape[1]), axis=0)

  # identify the indices corresponding to the start of each sample and set the corresponding rows to zeros
  ids_to_zero = np.arange(0, (n_samples * samples_len), samples_len)
  delta[ids_to_zero, :] = 0

  return delta


def delta_to_contig_pred(delta_pred, delta_true, n_samples, samples_len, window_size):

  """Converts 'delta' predictions to 'contiguous' predictions.

  @type delta_pred: numpy.ndarray
  @param delta_pred: NumPy 2D tensor containing delta predictions.
  @type delta_true: numpy.ndarray
  @param delta_true: NumPy 2D tensor containing delta ground truth.
  @type n_samples: int
  @param n_samples: number of samples (i.e., of separated data acquisitions).
  @type samples_len: int
  @param samples_len: length (i.e., number of elements) of each sample.
  @type window_size: int
  @param window_size: size of the window used for prediction.
  @rtype: Tuple[numpy.ndarray, numpy.ndarray]
  @returns: A tuple containing two NumPy 2D tensors: the contiguous predictions and the contiguous ground truth.
  """

  # convert delta ground truth to contiguous ground truth
  contig_true = _delta_to_contig(delta_true, n_samples, samples_len)

  # create a copy of delta predictions
  delta_pred_tmp = np.copy(delta_pred)

  # iterate over each sample
  for i in range(0, n_samples):

    # compute the lower bound indices for predictions and ground truth
    t_pred_lb = (samples_len - window_size) * i
    t_true_lb = (samples_len) * i + window_size

    # set the initial predicted delta values to be equal to the corresponding ground truth values
    delta_pred_tmp[t_pred_lb,:] = contig_true[t_true_lb,:]

  # convert modified delta predictions to contiguous predictions
  contig_pred = _delta_to_contig(delta_pred_tmp, n_samples, samples_len - window_size)

  return contig_pred, contig_true


def _delta_to_contig(delta, n_samples, samples_len):

  """Converts 'delta' measurements to 'contiguous' measurements.

  @type delta: numpy.ndarray
  @param delta: NumPy 2D tensor containing delta measurements.
  @type n_samples: int
  @param n_samples: number of samples (i.e., of separated data acquisitions).
  @type samples_len: int
  @param samples_len: length (i.e., number of elements) of each sample.
  @rtype: numpy.ndarray
  @returns: NumPy 2D tensor containing contiguous measurements.
  """

  # initialize an empty list to store contiguous measurements
  contig = []

  # iterate over each sample
  for i in range(0, n_samples):

    # compute the lower bound (inclusive) and upper bound (exclusive) indices for the current sample
    t_lb = samples_len * (i)
    t_ub = samples_len * (i + 1)

    # append the first measurement of the current sample to the contig list
    contig.append(delta[t_lb,:].tolist())

    # iterate over each subsequent time step within the current sample
    for t in range(t_lb + 1, t_ub):
      # calculate the current contiguous measurement by adding the delta measurement to the previous contiguous measurement
      contig.append((delta[t,:] + contig[-1]).tolist())

  # convert the contig list to a NumPy array
  contig = np.array(contig)

  return contig


def fit_dataset_scalers(dataset, scale_bounds):

  """Fits scalers for each feature in the dataset based on the given scale bounds.

  @type dataset: numpy.ndarray
  @param dataset: NumPy 2D tensor representing the dataset.
  @type scale_bounds: list
  @param scale_bounds: List of tuples containing the scale bounds for each feature.
  @rtype: list
  @returns: List of fitted scalers for each feature.
  """

  # initialize an empty list to store fitted scalers
  fitted_scalers = []

  # iterate over each feature
  for i in range(len(scale_bounds)):
    # fit a scaler for the current feature and append it to the fitted_scalers list
    fitted_scalers.append(_fit_scaler(dataset[:,i], scale_bounds[i]))

  return fitted_scalers


def _fit_scaler(data, feat_bounds):

  """Fits a scaler to the given data based on the provided feature bounds.

  @type data: numpy.ndarray
  @param data: NumPy 1D or 2D tensor representing the data.
  @type feat_bounds: tuple
  @param feat_bounds: Tuple containing the lower and upper bounds for the feature.
  @rtype: object
  @returns: Fitted scaler object.
  """

  # check if the data is 1D
  one_dim_data = (data.ndim == 1)

  if (one_dim_data):
    # reshape 1D data to a column vector
    data = data.reshape(-1,1)

  # create a MinMaxScaler object with the specified feature range
  scaler = MinMaxScaler(feature_range=(feat_bounds[0], feat_bounds[1]))

  # fit the scaler to the data
  fitted_scaler = scaler.fit(data)
    
  return fitted_scaler


def scale_dataset(dataset, scalers, inverse_transform=False):

  """Scales the dataset using the provided scalers.

  @type dataset: numpy.ndarray
  @param dataset: NumPy 1D or 2D tensor representing the dataset.
  @type scalers: list
  @param scalers: List of fitted scalers for each feature.
  @type inverse_transform: bool
  @param inverse_transform: Flag indicating whether to perform inverse scaling.
  @rtype: numpy.ndarray
  @returns: Scaled dataset.
  """

  # check if the dataset is 1D
  one_dim_data = (dataset.ndim == 1)

  if (one_dim_data):
    # reshape 1D dataset to a column vector
    dataset = dataset.reshape(-1,1)

  # create an empty 2D tensor to store the scaled dataset
  dataset_scaled = np.empty([dataset.shape[0], dataset.shape[1]])

  # iterate over each feature
  for i in range(len(scalers)):
    # scale the current feature of the dataset using the corresponding scaler
    dataset_scaled[:,i] = _scale_data(dataset[:,i], scalers[i], inverse_transform)

  if (one_dim_data):
    # reshape the scaled dataset back to 1D if the original dataset was 1D
    dataset_scaled = dataset_scaled.reshape(-1,)

  return dataset_scaled


def _scale_data(data, scaler, inverse_transform):

  """Scales the given data using the provided scaler.

  @type data: numpy.ndarray
  @param data: NumPy 1D or 2D tensor representing the data.
  @type scaler: object
  @param scaler: Fitted scaler object.
  @type inverse_transform: bool
  @param inverse_transform: Flag indicating whether to perform inverse scaling.
  @rtype: numpy.ndarray
  @returns: Scaled data.
  """

  # check if the data is 1D
  one_dim_data = (data.ndim == 1)

  if (one_dim_data):
    # reshape 1D data to a column vector
    data = data.reshape(-1,1)

  if (inverse_transform):
    # perform inverse scaling if the flag is True
    scaled_data = scaler.inverse_transform(data)
  else:
    # otherwise, perform normal scaling
    scaled_data = scaler.transform(data)

  if (one_dim_data):
    # reshape the scaled data back to 1D if the original data was 1D
    scaled_data = scaled_data.reshape(-1,)

  return scaled_data

