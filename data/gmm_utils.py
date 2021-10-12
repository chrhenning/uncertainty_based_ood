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
# @title          :data/gmm_utils.py
# @author         :fd, ch
# @contact        :henningc@ethz.ch
# @created        :07/07/2021
# @version        :1.0
# @python_version :3.8.10
"""
Utilitiy functions for GMM data
-------------------------------

Helper functions regarding the creation of data handlers using class
:class:`hypnettorch.data.special.gmm_data.GMMData`.
"""
from hypnettorch.data.special.gaussian_mixture_data import get_gmm_tasks
from hypnettorch.data.special.gmm_data import GMMData
import numpy as np

def circle_points_2d(r=5, n=6, offset=0):
    """Distribute points equally spaced on a circle.

    Args:
        r (int): Radius of the circle.
        n (int): Number of points on 2D circle.
        offset (float): Angular offset (rotates location of first point).

    Return:
        (numpy.ndarray): 2D-array containing ``n`` 2D points.
    """
    t = np.linspace(np.pi/2+offset, 5/2*np.pi+offset, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t)

    return np.c_[x, y]

def get_circle_gmm_instance(sigmas=(1., 1., 1.), num_train=10, num_test=100,
                            use_one_hot=True, radius=5, offset=0, rseed=42):
    """Create a GMM data handler with means positioned on a circle.

    Args:
        (....): See docstring of function
            :func:`hypnettorch.data.special.gaussian_mixture_data.\
get_gmm_tasks`.
        sigmas (list or tuple): A list of scalar variances :math:`\sigma`. The
            length of this list will determine the number of modes. Each mode
            is a 2D Gaussian :math:`\\mathcal{N}(\\vec{\mu}, \sigma I)` with
            means positioned around a circle.
        use_one_hot (bool): Whether the class labels should be represented as a
            one-hot encoding.
        radius (float): Radius of the circle.
        offset (float): Angular offset (rotates location of first mean).

    Returns:
        (hypnettorch.data.special.gmm_data.GMMData): The GMM data handler.
    """
    means = circle_points_2d(r=radius, n=len(sigmas), offset=offset)
    covs = [s * np.eye(2) for s in sigmas]

    modes = get_gmm_tasks(means=means, covs=covs, num_train=num_train,
                 num_test=num_test, map_functions=None, rseed=rseed)

    return GMMData(modes, classification=True, use_one_hot=use_one_hot,
                   mixing_coefficients=None)


if __name__ == '__main__':
    pass


