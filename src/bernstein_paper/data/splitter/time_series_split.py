#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : time_series_split.py
# brief  : [Description]
#
# author : Marcel Arpogaus
# date   : 2020-04-15 10:22:10
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
# modified time : 2020-04-15 17:53:59
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-15 10:22:10
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import numpy as np
import pandas as pd


class TimeSeriesSplit():
    RIGHT = 0
    LEFT = 1

    def __init__(self, split_size, split):
        self.split_size = split_size
        self.split = split

    def __call__(self, data):
        days = pd.date_range(data.index.min(), data.index.max(), freq='D')
        days = days.to_numpy().astype('datetime64[m]')
        right = days[int(len(days) * self.split_size)]
        left = right - 1
        if self.split == self.LEFT:
            return data.loc[:str(left)]
        else:
            return data.loc[str(right):]

