#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : windowed_time_series_pipeline.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-15 10:25:49
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
# modified time : 2020-05-18 10:13:11
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-15 10:25:49
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf


class PatchGenerator():

    def __init__(self, window_size, shift):
        self.window_size = window_size
        self.shift = shift

    def __call__(self, load_data):
        def sub_to_patch(sub):
            return sub.batch(self.window_size, drop_remainder=True)

        data_set = tf.data.Dataset.from_tensor_slices(load_data)

        data_set = data_set.window(
            size=self.window_size,
            shift=self.shift,
            # stride=1,
            drop_remainder=True
        )
        data_set = data_set.flat_map(sub_to_patch)
        # data_set = data_set.map(process_patch)

        return data_set


class BatchPreprocessor():

    def __init__(self, history_size,
                 historic_columns,
                 horizon_columns,
                 prediction_columns):
        self.history_size = history_size
        self.historic_columns = historic_columns
        self.horizon_columns = horizon_columns
        self.prediction_columns = prediction_columns

        columns = sorted(
            list(set(historic_columns + prediction_columns + horizon_columns)))
        self.column_idx = {c: i for i, c in enumerate(columns)}

    def __call__(self, batch):
        y = tf.stack([batch[:, self.history_size:, self.column_idx[c]]
                      for c in sorted(self.prediction_columns)], axis=2)

        x_hist = []
        x_hori = []
        x_columns = sorted(set(self.historic_columns + self.horizon_columns))
        if len(x_columns) == 0:
            ValueError('No feature columns provided')

        for ch in x_columns:
            p = batch[:, :self.history_size, self.column_idx[ch]]
            if ch in self.historic_columns:
                x_hist.append(p)
            else:
                x_hist.append(tf.zeros_like(p))
            p = batch[:, self.history_size:, self.column_idx[ch]]
            if ch in self.horizon_columns:
                x_hori.append(p)
            else:
                x_hori.append(tf.zeros_like(p))

        x_hist = tf.stack(x_hist, axis=2)
        x_hori = tf.stack(x_hori, axis=2)

        if len(self.historic_columns) == 0:
            x = x_hori
        elif len(self.horizon_columns) == 0:
            x = x_hist
        else:
            x = tf.concat([x_hist, x_hori], axis=1)

        return x, y


class WindowedTimeSeriesPipeline():
    def __init__(self,
                 history_size,
                 horizon_size,
                 historic_columns,
                 horizon_columns,
                 prediction_columns,
                 shift,
                 batch_size,
                 cycle_length,
                 shuffle_buffer_size,
                 seed):
        self.history_size = history_size
        self.horizon_size = horizon_size
        self.window_size = history_size + horizon_size
        self.historic_columns = historic_columns
        self.horizon_columns = horizon_columns
        self.prediction_columns = prediction_columns
        self.shift = shift
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __call__(self, ds):

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(self.cycle_length * self.shuffle_buffer_size,
                            seed=self.seed)

        ds = ds.interleave(
            PatchGenerator(self.window_size, self.shift),
            cycle_length=self.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(
                self.batch_size * self.shuffle_buffer_size,
                seed=self.seed)

        ds = ds.batch(self.batch_size)

        ds = ds.map(BatchPreprocessor(self.history_size,
                                      self.historic_columns,
                                      self.horizon_columns,
                                      self.prediction_columns),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.prefetch(1)
        ds = ds.cache()

        return ds
