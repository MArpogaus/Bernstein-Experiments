#!env python3
# AUTHOR INFORMATION ##########################################################
# file   : time_series_k_fold.py
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
# modified time : 2020-04-15 10:24:35
#  changes made : ...
# modified by   : Marcel Arpogaus
# modified time : 2020-04-15 10:22:10
#  changes made : newly written
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import numpy as np
import pandas as pd


class TimeSeriesKFold():
    def __init__(self, fold, n_folds=20):
        self.n_folds = n_folds
        self.fold = fold

    def __call__(self, data):
        days = pd.date_range(data.index.min(), data.index.max(), freq='D')
        fold_idx = np.array_split(days.to_numpy(), self.n_folds)
        folds = {f: (idx[[0, -1]].astype('datetime64[m]') + [0, 60 * 24 - 1]
                     ).astype(str).tolist() for f, idx in enumerate(fold_idx)}

        return data[folds[self.fold][0]:folds[self.fold][1]]
