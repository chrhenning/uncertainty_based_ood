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
# @title          :nngp/standard_kernels.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :11/15/2021
# @version        :1.0
# @python_version :3.8.12
"""
Standard kernel functions for Gaussian Processes
------------------------------------------------
"""
import torch

def rbf(X, length_scale=1.):
    r"""Compute the RBF kernel.

    .. math::

        k(x, x') = \exp \bigg( - \frac{\lVert x- x' \rVert_2^2}{2l^2} \bigg)

    Args:
        (....): See docstring of function :func:`nngp.mlp_kernel.init_kernel`.
        length_scale (float): Length-scale.
    
    Returns:
        (torch.Tensor): If ``X`` is passed as a single tensor, the symmetric
        kernel matrix (2D tensor) is returned. Otherwise, a vector of kernel
        values is returned.
    """
    if isinstance(X, torch.Tensor):
        X1 = X
        X2 = None
    else:
        X1 = X[0]
        X2 = X[1]

    assert X1.ndim in [1, 2]
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
        if X2 is not None:
            X2 = X2.reshape(-1, 1)

    if X2 is None:
        X_diff = X1.unsqueeze(1) - X1.unsqueeze(0)
        ed_diff = torch.sum(X_diff**2, dim=2)
    else:
        X_diff = X1 - X2
        ed_diff = torch.sum(X_diff**2, dim=1)

    return torch.exp(- 0.5 * ed_diff / length_scale**2)

if __name__ == '__main__':
    pass


