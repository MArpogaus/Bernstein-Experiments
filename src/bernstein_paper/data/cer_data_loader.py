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
