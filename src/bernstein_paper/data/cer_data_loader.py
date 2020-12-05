import os

import tensorflow as tf

from functools import partial

from .dataset import WindowedTimeSeriesDataSet
from .splitter import TimeSeriesSplit

def load_data(data_path: str,
              history_size,
              prediction_size,
              history_columns=['load', 'is_holiday', 'tempC'],
              meta_columns=['is_holiday', 'weekday'],
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
    :param      prediction_size:      The number of time steps in the
                                      prediction horizon a step should contain
    :type       prediction_size:      int
    :param      history_columns:     The historic columns
    :type       history_columns:     Array
    :param      meta_columns:         The column names to be used as horizon
                                      data.
    :type       meta_columns:         Array
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
    load_max = 14.134
    eps = 1e-5

    column_transformers = {}
    # column_transformers['load'] = lambda x: tf.math.log(
    #     (x + eps) / eps) / (tf.math.log(load_max / eps))
    column_transformers['load'] = lambda x: tf.sqrt(x / 14.134)
    column_transformers['weekday'] = lambda x: tf.one_hot(
        tf.cast(tf.squeeze(x), tf.uint8), 6)

    make_dataset = partial(WindowedTimeSeriesDataSet,
                           history_size=history_size,
                           prediction_size=prediction_size,
                           history_columns=history_columns,
                           meta_columns=meta_columns,
                           prediction_columns=prediction_columns,
                           shift=shift,
                           batch_size=32,
                           cycle_length=cycle_length,
                           shuffle_buffer_size=shuffle_buffer_size,
                           seed=seed,
                           column_transformers=column_transformers)

    # train data ##############################################################
    train_data_path = os.path.join(data_path, 'train.csv')
    test_data_path = os.path.join(data_path, 'test.csv')

    if 'train' in splits:
        if validation_split is not None:
            data_splitter = TimeSeriesSplit(
                1 - validation_split, TimeSeriesSplit.LEFT)
        else:
            data_splitter = None

        data['train'] = make_dataset(
            file_path=train_data_path,
            data_splitter=data_splitter)()

    # validation data #########################################################
    if 'validate' in splits and validation_split is not None:
        data_splitter = TimeSeriesSplit(
            validation_split, TimeSeriesSplit.RIGHT)

        data['validate'] = make_dataset(
            file_path=train_data_path,
            data_splitter=data_splitter)()

    # test data ###############################################################
    if 'test' in splits:
        data['test'] = make_dataset(
            file_path=test_data_path)()

    return data
