#!env python3
# -*- coding: utf-8 -*-

import os
import time
import pytest

import pandas as pd

from tqdm import trange
from tqdm import tqdm

from thesis.data.cer_data_loader import load_data

columns = ['load',
           'tempC',
           'is_holiday']

ids = list(range(100))
days = 30

period_range = pd.period_range('2001-01', periods=days * 48, freq='30min')

history_size = 48 * 3
horizon_size = 48
shift = horizon_size
historic_columns = ['load', 'is_holiday', 'tempC']
horizon_columns = ['is_holiday', 'tempC']
prediction_columns = ['load']
validation_split = 0.9


def benchmark(data, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in trange(num_epochs):
        epoch_start_time = time.perf_counter()
        for x, y in tqdm(data['train']):
            # Performing a training step
            time.sleep(0.001)
        print(f"Epoch {epoch_num} execution time:",
              time.perf_counter() - epoch_start_time)
    print("Execution time:", time.perf_counter() - start_time)


@pytest.fixture(scope="session")
def artificial_data(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("valid_data")
    dfs = []
    for i in ids:
        df = pd.DataFrame({'date_time': period_range,
                           'id': i,
                           **{c: n for n, c in enumerate(columns)}})
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.sort_values(['date_time', 'id'], inplace=True)
    dfs.to_csv(tmpdir / 'test.csv', index=False)
    dfs.to_csv(tmpdir / 'train.csv', index=False)
    return tmpdir


def test_artificial_data(artificial_data):

    data = load_data(
        data_path=artificial_data,
        history_size=history_size,
        horizon_size=horizon_size,
        historic_columns=historic_columns,
        prediction_columns=prediction_columns,
        horizon_columns=horizon_columns,
        shift=shift,
        validation_split=validation_split)

    benchmark(data)


def test_load_data_real():
    data_path = '../data/CER Electricity Revised March 2012/preprocessed'
    data = load_data(
        data_path=data_path,
        history_size=history_size,
        horizon_size=horizon_size,
        historic_columns=historic_columns,
        prediction_columns=prediction_columns,
        horizon_columns=horizon_columns,
        shift=shift,
        validation_split=validation_split)

    benchmark(data)
