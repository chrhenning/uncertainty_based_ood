#!/usr/bin/env python3
# Copyright 2021 Christian Henning
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
#
# @title          :data/regression_utils.py
# @author         :fd, ch
# @contact        :henningc@ethz.ch
# @created        :09/15/2021
# @version        :1.0
# @python_version :3.8.10
"""
Helper functions for working with regression datasets
-----------------------------------------------------
"""
from hypnettorch.data.special.regression1d_data import ToyRegression
import numpy as np

def generate_1d_dataset(task_id=0, num_train=20, num_test=100, num_val=None,
                        rseed=42):
    """Generate a set of tasks for 1D regression.

    Args:
        (....) See docstring of class
            :class:`hypnettorch.data.special.regression1d_data.ToyRegression`.
        task_id (int): Determines the regression task to be generated.

    Returns:
        A data handler.
    """
    map_funcs = [
        lambda x: (x) ** 3.,
        lambda x: (3.*x),
        lambda x: 2. * np.power(x, 2) - 1,
        lambda x: np.power(x - 3., 3),
        lambda x: x*np.sin(x),
        lambda x: 0.3*x*(1+np.sin(x)),
        lambda x: 2*np.sin(x)+np.sin(np.sqrt(2)*x)+np.sin(np.sqrt(3)*x)
    ]

    train_domains = [
        [-3.5, 3.5],
        [-2, 2],
        [-1, 1],
        [2, 4],
        [[-0.5,0.5], [1.5,2.5], [4.5,6.0], [9,11]],
        [[4.5,5], [7.5,8.5], [10,11]],
        [[1.0,1.3], [3.5,3.8], [5.2,5.5]]
    ]

    test_domains = [[-5.0, 5.0], [-3, +3], [-2.5, 2.5], [.5, 4.1], [-3, 13],
                    [-1, 12], [-0.5, 7.0]]

    val_domains = [
        [-3.5, 3.5],
        [-2, 2],
        [-1, 1],
        [2, 4],
        [0, 10],
        [3, 12],
        [-0.5, 7.0]
    ]

    std = [3, 0.05, 0.05, 0.05, 0.25, 0.6, 0.2]

    data = ToyRegression(train_inter=train_domains[task_id],
                         num_train=num_train, test_inter=test_domains[task_id],
                         num_test=num_test,
                         val_inter=None if num_val is None else \
                             val_domains[task_id], num_val=num_val,
                         map_function=map_funcs[task_id], std=std[task_id],
                         rseed=rseed)

    return data

if __name__ == '__main__':
    pass


