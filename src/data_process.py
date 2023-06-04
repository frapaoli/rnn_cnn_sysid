# data analysis and manipulation
from sklearn.preprocessing import MinMaxScaler

# tensor operations
import numpy as np

def contig_to_delta_meas(contig_meas, n_samples, samples_len):
  """
  Given a tensor of contiguous time-varying measurements, returns a tensor representing the 'delta' of those measurements
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

  delta_meas = np.empty([contig_meas.shape[0], contig_meas.shape[1]])
  delta_meas[:,0]  = contig_meas[:,0]
  delta_meas[:,1:] = _contig_to_delta(contig_meas[:,1:], n_samples, samples_len)

  return delta_meas


def _contig_to_delta(contig, n_samples, samples_len):

  delta  = np.diff(contig, axis=0)
  delta  = np.insert(delta, 0, np.zeros(delta.shape[1]), axis=0)
  ids_to_zero = np.arange(0, (n_samples * samples_len), samples_len)
  delta[ids_to_zero, :] = 0

  return delta


def delta_to_contig_pred(delta_pred, delta_true, n_samples, samples_len, window_size):

  contig_true = _delta_to_contig(delta_true, n_samples, samples_len)

  delta_pred_tmp = np.copy(delta_pred)

  for i in range(0, n_samples):

    t_pred_lb = (samples_len - window_size) * i
    t_true_lb = (samples_len) * i + window_size

    delta_pred_tmp[t_pred_lb,:] = contig_true[t_true_lb,:]

  contig_pred = _delta_to_contig(delta_pred_tmp, n_samples, samples_len - window_size)

  return contig_pred, contig_true


def _delta_to_contig(delta, n_samples, samples_len):

  contig = []

  for i in range(0, n_samples):

    t_lb = samples_len * (i)
    t_ub = samples_len * (i + 1)

    contig.append(delta[t_lb,:].tolist())

    for t in range(t_lb + 1, t_ub):
      contig.append((delta[t,:] + contig[-1]).tolist())

  contig = np.array(contig)
  return contig


def fit_dataset_scalers(dataset, scale_bounds):

  fitted_scalers = []
  for i in range(len(scale_bounds)):
    fitted_scalers.append(_fit_scaler(dataset[:,i], scale_bounds[i]))

  return fitted_scalers


def _fit_scaler(data, feat_bounds):

  one_dim_data = (data.ndim == 1)

  if (one_dim_data):
    data = data.reshape(-1,1)

  scaler = MinMaxScaler(feature_range=(feat_bounds[0], feat_bounds[1]))
  fitted_scaler = scaler.fit(data)
    
  return fitted_scaler


def scale_dataset(dataset, scalers, inverse_transform=False):

  one_dim_data = (dataset.ndim == 1)

  if (one_dim_data):
    dataset = dataset.reshape(-1,1)

  dataset_scaled = np.empty([dataset.shape[0], dataset.shape[1]])

  for i in range(len(scalers)):
    dataset_scaled[:,i] = _scale_data(dataset[:,i], scalers[i], inverse_transform)

  if (one_dim_data):
    dataset_scaled = dataset_scaled.reshape(-1,)

  return dataset_scaled


def _scale_data(data, scaler, inverse_transform):

  one_dim_data = (data.ndim == 1)

  if (one_dim_data):
    data = data.reshape(-1,1)

  if (inverse_transform):
    scaled_data = scaler.inverse_transform(data)
  else:
    scaled_data = scaler.transform(data)

  if (one_dim_data):
    scaled_data = scaled_data.reshape(-1,)

  return scaled_data