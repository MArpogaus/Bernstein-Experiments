import os

from functools import partial

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

from .dataset import WindowedTimeSeriesDataSet
from .splitter import TimeSeriesSplit


def load_data(data_path: str,
              history_size,
              horizon_size,
              historic_columns=['load', 'is_holiday', 'tempC'],
              horizon_columns=['is_holiday', 'tempC'],
              prediction_columns=['load'],
              splits=['train', 'validate', 'test'],
              shift=None,
              validation_split=None,
              batch_size=32,
              cycle_length=10,
              shuffle_buffer_size=1000,
              seed=42):
    """
    Loads the preprocessed CER data and build the dataset.

    :param      data_path:            The path to the folder containing the
                                      train.csv and test.csv
    :type       data_path:            str
    :param      history_size:         The number of time steps of the historic
                                      data a patch should contain
    :type       history_size:         int
    :param      horizon_size:         The number of time steps in the
                                      prediction horizon a step should contain
    :type       horizon_size:         int
    :param      historic_columns:     The column names to used as historic
                                      data.
    :type       historic_columns:     Array
    :param      horizon_columns:      The column names to be used as horizon
                                      data.
    :type       horizon_columns:      Array
    :param      prediction_columns:   The columns to predict
    :type       prediction_columns:   Array
    :param      splits:               The data splits to be generated. At least
                                      one of 'train', 'validate' or 'test'
    :type       splits:               Array
    :param      shift:                The amount of time steps by which the
                                      window moves on each iteration
    :type       shift:                int
    :param      validation_split:     The amount of data reserved from the
                                      training set for validation
    :type       validation_split:     float
    :param      batch_size:           The batch size
    :type       batch_size:           int
    :param      cycle_length:         The number of input elements that are
                                      processed concurrently
    :type       cycle_length:         int
    :param      shuffle_buffer_size:  The shuffle buffer size
    :type       shuffle_buffer_size:  int
    :param      seed:                 The seed used by the pseudo random
                                      generators
    :type       seed:                 int

    :returns:   A dict containing the windowed TensorFlow datasets generated
                from csv file in `data_path` for the given `spits`.
    :rtype:     dict
    """

    # common ##################################################################
    data = {}

    scalers = {
        'load': MinMaxScaler(feature_range=(0, 1)),
        'tempC': MinMaxScaler(feature_range=(-1, 1)),
        'is_holiday': MinMaxScaler(feature_range=(0, 1))
    }

    column_transformer = make_column_transformer(
        *[(scalers[k], [k]) for k in sorted(scalers.keys())])

    make_dataset = partial(WindowedTimeSeriesDataSet,
                           column_transformer=column_transformer,
                           history_size=history_size,
                           horizon_size=horizon_size,
                           historic_columns=historic_columns,
                           horizon_columns=horizon_columns,
                           prediction_columns=prediction_columns,
                           shift=shift,
                           batch_size=32,
                           cycle_length=cycle_length,
                           shuffle_buffer_size=shuffle_buffer_size,
                           seed=seed)

    # train data ##############################################################
    if 'train' in splits:
        if validation_split is not None:
            data_splitter = TimeSeriesSplit(
                1 - validation_split, TimeSeriesSplit.LEFT)
        else:
            data_splitter = None
        train_data_path = os.path.join(data_path, 'train.csv')

        data['train'] = make_dataset(file_path=train_data_path,
                                     data_splitter=data_splitter,
                                     fit_transformer=True)()

    # validation data #########################################################
    if 'validate' in splits and validation_split is not None:
        data_splitter = TimeSeriesSplit(
            validation_split, TimeSeriesSplit.RIGHT)

        data['validate'] = make_dataset(file_path=train_data_path,
                                        data_splitter=data_splitter)()

    # test data ###############################################################
    if 'test' in splits:
        test_data_path = os.path.join(data_path, 'test.csv')
        data['test'] = make_dataset(file_path=test_data_path)()

    return data
