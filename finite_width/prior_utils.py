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
# @title          :finite_width/prior_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :05/22/2021
# @version        :1.0
# @python_version :3.8.8
"""
Prior distributions for finite-width neural networks
----------------------------------------------------

This module contains a couple of helper functions regarding weight-space prior
distributions.
"""
from hypnettorch.mnets import MainNetInterface
import torch
from torch.distributions.normal import Normal

def width_aware_gaussian_prior(net, w_var=1., b_var=1., device=None):
    r"""Gaussian prior with width-corrected variance.

    This function generates a Gaussian prior with diagonal covariance matrix,
    where bias vectors are following a distribution
    :math:`\mathcal{N}(0, \sigma_b^2)` and layer weights follow a distribution
    :math:`\mathcal{N}(0, \sigma_w^2/N_l)` where :math:`N_l` is the width of
    layer :math:`l`.

    Importantly, variances are normalized using a fan-in criterion, such that
    the input variance is propagated to the output units.

    Args:
        net (hypnettorch.mnets.mnet_interface.MainNetInterface): The network
            for which the prior should be created. The prior is generated for
            all weights according to the network attribute ``param_shapes``.
        w_var (float): The weight variance :math:`\sigma_w^2`.
        b_var (float): The bias variance :math:`\sigma_w^2`.
        device: PyTorch device.

    Returns:
        (tuple): Tuple containing:
          - **means** (list): List of tensors comprising the mean for each
            parameter according to attribute ``param_shapes``. Note, all means
            are zero.
          - **vars** (list): List of tensors comprising the variance for each
            parameter according to attribute ``param_shapes``.

    Function :func:`to_normal_dist` can be used to transform those values into
    an instance of class :class:`torch.distributions.normal.Normal`.
    """
    assert isinstance(net, MainNetInterface)

    if device is None:
        if net.internal_params is not None:
            device = net.internal_params[0].device
        else:
            device = 'cpu'

    meta = None
    try:
        meta = net.param_shapes_meta
    except:
        pass

    means = []
    variances = []

    for i, s in enumerate(net.param_shapes):
        # Note, the first two entries of a shape tuple are
        # `[fan_out, fan_in, ...]`. So, `s[1]` is the fan-in.
        v = -1
        if meta is None:
            v = w_var / s[1] if len(s) > 1 else b_var
        else:
            if meta[i]['name'] == 'weight':
                v = w_var / s[1]
            else: # All other types of parameters are considered bias for now.
                v = b_var

        means.append(torch.zeros(*s).to(device))
        variances.append(torch.ones(*s).to(device) * v)

    return means, variances

def to_normal_dist(means, variances, flattened=False):
    """Convert distributional parameters to distribution.

    Args:
        means (list or torch.Tensor): List of mean tensors or single mean
            tensor.
        variances (list or torch.Tensor): List of variance tensors or single
            variance tensor.
        flattened (bool): If ``True`` and ``means`` is a list, then the tensors
            in ``means`` and ``variances`` are first flattened and
            concatenated, and a single distribution object is returned.

    Returns:
        (list or torch.distributions.normal.Normal)
    """
    if not isinstance(means, torch.Tensor) or flattened:
        if not isinstance(means, torch.Tensor):
            means = MainNetInterface.flatten_params(means)
            variances = MainNetInterface.flatten_params(variances)

            return Normal(means, torch.sqrt(variances))

    assert len(means) == len(variances)
    ret = []
    for i, m in enumerate(means):
        v = variances[i]

        ret.append(Normal(m, torch.sqrt(v)))

    return ret



if __name__ == '__main__':
    pass


