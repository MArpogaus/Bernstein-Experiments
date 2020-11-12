#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : windowed_time_series_data_set.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-15 10:28:57
# COPYRIGHT ###################################################################
# Copyright 2020 Marcel Arpogaus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTES ######################################################################
#
# This project is following the
# [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
#
# CHANGELOG ##################################################################
# modified by   : Marcel Arpogaus
# modified time : 2020-04-15 20:46:28
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-15 10:28:57
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf
import numpy as np

from ..loader import CSVDataLoader
from ..pipeline import WindowedTimeSeriesPipeline


# ref.: https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
def encode(data, cycl_name, cycl=None):
    if cycl is None:
        cycl = getattr(data.index, cycl_name)
    cycl_max = cycl.max()
    data[cycl_name + '_sin'] = np.float32(np.sin(2 * np.pi * cycl / cycl_max))
    data[cycl_name + '_cos'] = np.float32(np.cos(2 * np.pi * cycl / cycl_max))
    return data


class DatasetGenerator():
    def __init__(self, df, columns):
        self.grpd = df.groupby('id')
        self.columns = columns

    def __call__(self):
        for _, d in self.grpd:
            yield d[self.columns].values


class WindowedTimeSeriesDataSet():
    def __init__(self,
                 file_path,
                 history_size,
                 prediction_size,
                 history_columns,
                 meta_columns,
                 prediction_columns,
                 data_splitter=None,
                 column_transformers={},
                 shift=None,
                 batch_size=32,
                 cycle_length=100,
                 shuffle_buffer_size=1000,
                 seed=42):
        self.columns = sorted(list(
            set(history_columns + prediction_columns + meta_columns)))
        dtype = {'id': 'uint16',
                 'load': 'float32',
                 'is_holiday': 'uint8',
                 'weekday': 'uint8'}
        shift = shift or prediction_size

        self.data_loader = CSVDataLoader(
            file_path=file_path,
            dtype=dtype
        )
        self.data_splitter = data_splitter

        self.data_pipeline = WindowedTimeSeriesPipeline(
            history_size=history_size,
            prediction_size=prediction_size,
            history_columns=history_columns,
            meta_columns=meta_columns,
            prediction_columns=prediction_columns,
            shift=shift,
            batch_size=batch_size,
            cycle_length=cycle_length,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            column_transformers=column_transformers
        )

    def __call__(self):
        data = self.data_loader()
        data = encode(data, 'dayofyear')
        data = encode(data, 'time', data.index.hour * 60 + data.index.minute)
        if self.data_splitter is not None:
            data = self.data_splitter(data)

        generator = DatasetGenerator(data, self.columns)
        ds = tf.data.Dataset.from_generator(
            generator,
            output_types=tf.float32,
            output_shapes=tf.TensorShape([None, len(self.columns)]))
        ds = self.data_pipeline(ds)
        return ds
