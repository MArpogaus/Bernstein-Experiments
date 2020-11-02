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

from ..loader import CSVDataLoader
from ..pipeline import WindowedTimeSeriesPipeline


class DatasetGenerator():
    def __init__(self, df, columns, column_transformer=None):
        self.grpd = df.groupby('id')
        self.columns = columns
        self.column_transformer = column_transformer

    def __call__(self):
        for _, d in self.grpd:
            if self.column_transformer is not None:
                yield self.column_transformer.transform(d)
            else:
                yield d[self.columns].values


class WindowedTimeSeriesDataSet():
    def __init__(self,
                 file_path,
                 history_size,
                 horizon_size,
                 historic_columns,
                 horizon_columns,
                 prediction_columns,
                 data_splitter=None,
                 column_transformer=None,
                 shift=None,
                 batch_size=32,
                 cycle_length=100,
                 shuffle_buffer_size=1000,
                 fit_transformer=False,
                 seed=42):
        self.columns = sorted(list(
            set(historic_columns + prediction_columns + horizon_columns)))
        dtype = {'id': 'uint16',
                 'load': 'float32',
                 'tempC': 'int8',
                 'is_holiday': 'uint8'}
        shift = shift or horizon_size

        self.data_loader = CSVDataLoader(
            file_path=file_path,
            columns=self.columns + ['id'],
            dtype=dtype
        )
        self.data_splitter = data_splitter
        self.column_transformer = column_transformer
        self.data_pipeline = WindowedTimeSeriesPipeline(
            history_size=history_size,
            horizon_size=horizon_size,
            historic_columns=historic_columns,
            horizon_columns=horizon_columns,
            prediction_columns=prediction_columns,
            shift=shift,
            batch_size=batch_size,
            cycle_length=cycle_length,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed
        )
        self.fit_transformer = fit_transformer

    def __call__(self):
        data = self.data_loader()
        if self.data_splitter is not None:
            data = self.data_splitter(data)
        if self.column_transformer is not None and self.fit_transformer:
            self.column_transformer.fit(data)

        generator = DatasetGenerator(
            data, self.columns, self.column_transformer)
        ds = tf.data.Dataset.from_generator(
            generator,
            output_types=tf.float32,
            output_shapes=tf.TensorShape([None, len(self.columns)]))
        ds = self.data_pipeline(ds)
        return ds
