#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : csv_data_loader.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-15 10:20:37
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
# modified time : 2020-04-15 10:21:11
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-15 10:20:37
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import pandas as pd


def _read_csv_file(file_path, columns, **kwds):
    file_path = file_path
    load_data = pd.read_csv(file_path,
                            parse_dates=['date_time'],
                            infer_datetime_format=True,
                            index_col=['date_time'],
                            usecols=['date_time'] + columns,
                            **kwds)

    # merge with weather data
    if load_data.isnull().any().sum() != 0:
        raise ValueError('Data contains NaNs')

    return load_data


class CSVDataLoader():
    def __init__(self, file_path, columns, **kwds):
        self.file_path = file_path
        self.columns = columns
        self.kwds = kwds

    def __call__(self):
        return _read_csv_file(self.file_path,
                              self.columns,
                              **self.kwds)
