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
# @title          :utils/misc.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/09/2021
# @version        :1.0
# @python_version :3.8.10
"""
Miscellaneous Utility Functions
-------------------------------
"""
import numpy as np
import torch

def eval_grid_2d(x_range=(-10, 10), y_range=(-10, 10), res_per_unit=10,
                 verbose=False):
    """Generate grid of equispaced points in 2D-space.

    Args:
        x_range (tuple): Start and end position of the grid in x-range.
        y_range (tuple): Start and end position of the grid in y-range.
        res_per_unit (int): The resolution of the grid per unit.
        verbose (bool): Whether the grid resolution should be printed.

    Returns:
        (tuple): Tuple containing:

        - **grid_X** (numpy.ndarray): A 2D array of the grid, containing the
          x-value at every (x, y) position.
        - **grid_Y** (numpy.ndarray): Same as ``grid_X`` but for y-values.
        - **grid_XY** (numpy.ndarray): A 2D array of shape ``[N, 2]`` containing
          the (x,y)-value for each of the ``N`` grid-positions.
    """
    assert len(x_range) == 2 and len(y_range) == 2

    grid_size_x = int(np.ceil((x_range[1] - x_range[0]) * res_per_unit))
    grid_size_y = int(np.ceil((y_range[1] - y_range[0]) * res_per_unit))

    if verbose:
        print('Grid resolution: %d x %d.' % (grid_size_x, grid_size_y))

    grid_x = np.linspace(start=x_range[0], stop=x_range[1], num=grid_size_x)
    grid_y = np.linspace(start=y_range[0], stop=y_range[1], num=grid_size_y)

    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grid_XY = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

    return grid_X, grid_Y, grid_XY

def calc_regression_acc(predictions, targets, thld=None):
    """Calculate accuracy when treating regression as binary classification.

    Args:
        predictions (torch.Tensor): The predicted mean. A mean greater than zero
            will be considered a postive prediction.
        targets (torch.Tensor): The regression targets. Targets greater than
            zero will be considered as positive lables.
        thld (float, optional): If specified, another threshold than zero can be
            chosen to define the boundary between positive and negative values.

    Returns:
        Accuracy.
    """
    if thld is None:
        thld = 0
    labels = targets > thld
    pred_labels = predictions > thld

    return 100 * torch.sum(labels == pred_labels) / labels.shape[0]

if __name__ == '__main__':
    pass


