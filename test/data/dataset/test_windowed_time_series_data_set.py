#!env python3
# -*- coding: utf-8 -*-

import time
import pytest
import numpy as np

import pandas as pd

from tqdm import tqdm
from tqdm import trange

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler

from thesis.data.dataset import WindowedTimeSeriesDataSet
from thesis.data.splitter import TimeSeriesSplit

from thesis.util.visualization import plot_patches

columns = ['load',
           'tempC',
           'is_holiday']


x_hdim = 48
x_vdim = 7
y_hdim = 48
y_vdim = 1

history_size = x_hdim * x_vdim
horizon_size = y_hdim * y_vdim
shift = horizon_size

historic_columns = columns
horizon_columns = columns[1:]
prediction_columns = columns[:1]

batch_size = 32
cycle_length = 10
shuffle_buffer_size = 100

ids = list(range(10))
periods = (history_size + horizon_size) * 3

period_range = pd.period_range('2001-01', periods=periods, freq='30min')


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in trange(num_epochs):
        epoch_start_time = time.perf_counter()
        for _ in tqdm(dataset):
            # Performing a training step
            time.sleep(0.0001)
        print(f"Epoch {epoch_num} execution time:",
              time.perf_counter() - epoch_start_time)
    print("Execution time:", time.perf_counter() - start_time)


@pytest.fixture(scope="session")
def test_data(tmpdir_factory):
    file_path = tmpdir_factory.mktemp("csv_data") / 'test.csv'
    dfs = []
    for i in ids:
        df = pd.DataFrame({'date_time': period_range,
                           'id': i,
                           **{c: [int(f'{i:02d}{n:02d}{l:03d}') for l in range(periods)]
                              for n, c in enumerate(columns)}})
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv(file_path, index=False)

    return file_path


def test_artificial_data(test_data):

    data_set = WindowedTimeSeriesDataSet(
        file_path=test_data,
        history_size=history_size,
        horizon_size=horizon_size,
        historic_columns=historic_columns,
        horizon_columns=horizon_columns,
        prediction_columns=prediction_columns,
        shift=shift,
        batch_size=batch_size,
        cycle_length=cycle_length,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=42)
    test_data = data_set()
    print(test_data)
    print(next(test_data.take(1).as_numpy_iterator()))
    benchmark(test_data)


def test_artificial_data_column_transformer(test_data):
    scalers = {
        'load': MinMaxScaler(feature_range=(0, 1)),
        'tempC': MinMaxScaler(feature_range=(-1, 1)),
        'is_holiday': MinMaxScaler(feature_range=(0, 1))
    }

    column_transformer = make_column_transformer(
        *[(scalers[k], [k]) for k in sorted(scalers.keys())])

    data_splitter = TimeSeriesSplit(0.5, TimeSeriesSplit.LEFT)

    data_set = WindowedTimeSeriesDataSet(
        file_path=test_data,
        column_transformer=column_transformer,
        data_splitter=data_splitter,
        history_size=history_size,
        horizon_size=horizon_size,
        historic_columns=historic_columns,
        horizon_columns=horizon_columns,
        prediction_columns=prediction_columns,
        shift=shift,
        batch_size=32,
        cycle_length=10,
        shuffle_buffer_size=100,
        fit_transformer=True,
        seed=42)
    test_data = data_set()

    print(next(test_data.take(1).as_numpy_iterator()))
    benchmark(test_data)


def test_real_data():
    file_path = '../data/CER Electricity Revised March 2012/preprocessed/test.csv'

    data_set = WindowedTimeSeriesDataSet(
        file_path=file_path,
        history_size=history_size,
        horizon_size=horizon_size,
        historic_columns=historic_columns,
        horizon_columns=horizon_columns,
        prediction_columns=prediction_columns,
        shift=shift,
        batch_size=batch_size,
        cycle_length=cycle_length,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=42)
    test_data = data_set()
    print(next(test_data.take(1).as_numpy_iterator()))
    benchmark(test_data)


def test_load_data_viz():
    file_path = '../data/CER Electricity Revised March 2012/preprocessed/test.csv'
    data_set = WindowedTimeSeriesDataSet(
        file_path=file_path,
        history_size=history_size,
        horizon_size=horizon_size,
        historic_columns=historic_columns,
        horizon_columns=horizon_columns,
        prediction_columns=prediction_columns,
        shift=shift,
        batch_size=batch_size,
        cycle_length=1,
        shuffle_buffer_size=0,
        seed=42)
    test_data = data_set()

    N = 3

    height_ratios = [len(set(historic_columns + horizon_columns)) *
                     (np.where(len(historic_columns), x_vdim, 0) +
                      np.where(len(horizon_columns), y_vdim, 0)),
                     len(prediction_columns) * y_vdim]

    fig_height = sum(height_ratios) + min(len(historic_columns),
                                          1) + min(len(prediction_columns), 1)
    fig_height /= 5
    fig_width = N * (max(x_hdim, y_hdim) + min(N - 1, 1)) / 7

    plot_patches(test_data,
                 N=N,
                 x_hdim=x_hdim,
                 x_vdim=x_vdim,
                 y_hdim=y_hdim,
                 y_vdim=y_vdim,
                 historic_columns=historic_columns,
                 horizon_columns=horizon_columns,
                 prediction_columns=prediction_columns,
                 title_map={'x': 'Input Data',
                            'y': 'Prediction Target'},
                 y_label_map={
                     'x': {
                         'load': 'Load',
                         'tempC': 'Temperature',
                         'is_holiday': 'Is Holiday'},
                     'y': {
                         'load': 'Load'}},
                 fig_kw={'figsize': (fig_width, fig_height)},
                 heatmap_kw={
                     'x': {
                         'load': {'cmap': 'OrRd'},
                         'tempC': {'cmap': 'RdBu_r'},
                         'is_holiday': {'cmap': 'binary'}},
                     'y': {
                         'load': {'cmap': 'OrRd'}}},
                 gridspec_kw={'height_ratios': height_ratios,
                              'hspace': 2 / fig_height,
                              'wspace': 1 / fig_width},
                 xy_ch_connect=(
                     ('load', 0),
                     ('load', x_vdim + y_vdim - 1 - shift // x_hdim)))
