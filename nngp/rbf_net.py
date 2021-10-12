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
# @title          :nngp/rbf_net.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :06/18/2021
# @version        :1.0
# @python_version :3.8.10
r"""
The Kernel of infinite-width RBF networks
-----------------------------------------

The module :mod:`nngp.rbf_net` implements ways to approximate or compute the
induced kernel of an infinite-width RBF network.

The implementations in this module are inspired by the paper:

    `Williams <https://dl.acm.org/doi/10.5555/2998981.2999023>`__,
    Computing with Infinite Networks, 1996.

We use the following notation. The input is
:math:`\mathbf{x} \equiv \mathbf{x}^0`, and the output is
:math:`f(\mathbf{x}) \equiv \mathbf{x}^L`, where

.. math::

    \mathbf{x}^l = W^l \mathbf{z}^l + \mathbf{b}^l

with

.. math::

    \mathbf{z}^l = h(\mathbf{x}^{l-1}, \mathbf{u}^l) = \
        \exp \bigg( - \frac{1}{2 \sigma_g^2} \
        \lVert \mathbf{x}^{l-1} - \mathbf{u}^l  \rVert_2^2 \bigg)

Traditionally (and as reviewed by Williams), RBF networks have only 1 layer
(:math:`L = 1`). We are not aware of many works that consider multi-layer RBF
networks. There is, for instance, the work by
`Craddock et al. <https://ieeexplore.ieee.org/document/548981>`__, that defines
multi-layer RBF networks by considering each hidden unit is an individual RBF
network. Note, that this type of multi-layer RBF network can be considered a
special case of our definition above, where columns in :math:`W^l` only have one
non-zero entry.

To introduce the basic notation:
:math:`\mathbf{x}^l \in \mathbb{R}^{M_l}` with :math:`M_0 \equiv d_\text{in}`
and :math:`M_L \equiv d_\text{out}`,
:math:`\mathbf{z}^l \in \mathbb{R}^{N_l}` for :math:`0 < l \leq L`, and
:math:`\mathbf{u}^l \in \mathbb{R}^{M_{l-1}}` for :math:`0 < l \leq L`.
We assume that :math:`\mathbf{b}^l` and :math:`W^l` are drawn i.i.d. from a
zero-mean distribution with variance :math:`\sigma_b^2` and
:math:`\frac{\sigma_w^2}{N_l}`, respectively. In addition, we assume
:math:`\mathbf{u}^l \sim \mathcal{N}(0, \sigma_u^2 I)`.

When we refer to the infinite-width limit, we consider the case
:math:`N_l \rightarrow \infty`, while :math:`M_l` is assumed to be finite.

Assuming :math:`N_l \rightarrow \infty`, it follows that

.. math::

    \mathbb{E}[x^l_i] &= 0 \\ \
    \mathbb{E}[x^l_i(\mathbf{x}) x^l_i(\mathbf{x}')] &= \
        \sigma_b^2 + \sigma_w^2 \
        \mathbb{E}_{\mathbf{x}^{l-1}(\mathbf{x}), \
                    \mathbf{x}^{l-1}(\mathbf{x}'), \mathbf{u}^l} \
        \big[ h(\mathbf{x}^{l-1}(\mathbf{x}), \mathbf{u}^l) \
              h(\mathbf{x}^{l-1}(\mathbf{x}'), \mathbf{u}^l) \big]

Following the same argumentation as provided by Williams for a single layer
(cf. Sec. 3), it is easy to see that values :math:`x^l_i` are Gaussian
distributed in the infinite-width limit. Moreover, Williams analytically
derives the variance of :math:`x^1_i` as

.. math::

    k^1(\mathbf{x}, \mathbf{x}') &= \
        \mathbb{E}[x^1_i(\mathbf{x}) x^1_i(\mathbf{x}')] \\ \
        &= \sigma_b^2 + \sigma_w^2 \mathbb{E}_{\mathbf{u}^1} \
        \big[ h(\mathbf{x}, \mathbf{u}^1) \
              h(\mathbf{x}', \mathbf{u}^1) \big] \\ \
        &= \sigma_b^2 + \sigma_w^2 V_G(\mathbf{x}, \mathbf{x}', M_0)

where

.. math::
    :label: v_g_williams

    V_G(\mathbf{x}, \mathbf{x}', d) = \
        \bigg( \frac{\sigma_e}{\sigma_u} \bigg)^d \
        \exp \bigg( - \frac{\mathbf{x}^T \mathbf{x}}{2 \sigma_m^2} \bigg) \
        \exp \bigg( - \frac{(\mathbf{x}-\mathbf{x}')^T \
                            (\mathbf{x}-\mathbf{x}')}{2 \sigma_s^2} \bigg) \
        \exp \bigg( - \frac{\mathbf{x}'^T \mathbf{x}'}{2 \sigma_m^2} \bigg)

where :math:`1/\sigma_e^2 = 2/\sigma_g^2 + 1/\sigma_u^2`,
:math:`\sigma_s^2 = 2\sigma_g^2 + \sigma_g^4/\sigma_u^2` and
:math:`\sigma_m^2 = 2\sigma_u^2 + \sigma_g^2`.

We can use :math:`V_G(\cdot, \cdot, \cdot)` to recursively compute (or estimate)
the kernel :math:`k^l(\mathbf{x}, \mathbf{x}')`

.. math::
    :label: rbf-net-kernel-recursion

    k^l(\mathbf{x}, \mathbf{x}') &= \
        \mathbb{E}[x^l_i(\mathbf{x}) x^l_i(\mathbf{x}')] \\ \
        &= \sigma_b^2 + \sigma_w^2 \
        \mathbb{E}_{ \
            \begin{bmatrix} x_j^{l-1}(\mathbf{x}) \\ \
            x_j^{l-1}(\mathbf{x}') \end{bmatrix} \sim \mathcal{N} \bigg( \
            \mathbf{0}, \begin{bmatrix} \
                k^{l-1}(\mathbf{x}, \mathbf{x}) & \
                k^{l-1}(\mathbf{x}, \mathbf{x}') \\ \
                k^{l-1}(\mathbf{x}, \mathbf{x}') & \
                k^{l-1}(\mathbf{x}', \mathbf{x}') \
            \end{bmatrix} \bigg), j = 1, \cdots, M^{l-1}} \Big[ \
        \mathbb{E}_{\mathbf{u}^l} \big[ \
              h(\mathbf{x}^{l-1}(\mathbf{x}), \mathbf{u}^l) \
              h(\mathbf{x}^{l-1}(\mathbf{x}'), \mathbf{u}^l) \big] \Big] \\ \
        &= \sigma_b^2 + \sigma_w^2 \
        \mathbb{E}_{ \
            \begin{bmatrix} x_j^{l-1}(\mathbf{x}) \\ \
            x_j^{l-1}(\mathbf{x}') \end{bmatrix} \sim \mathcal{N} \bigg( \
            \mathbf{0}, \begin{bmatrix} \
                k^{l-1}(\mathbf{x}, \mathbf{x}) & \
                k^{l-1}(\mathbf{x}, \mathbf{x}') \\ \
                k^{l-1}(\mathbf{x}, \mathbf{x}') & \
                k^{l-1}(\mathbf{x}', \mathbf{x}') \
            \end{bmatrix} \bigg), j = 1, \cdots, M^{l-1}} \Big[ \
        V_G(\mathbf{x}^{l-1}(\mathbf{x}), \mathbf{x}^{l-1}(\mathbf{x}'), \
            M^{l-1}) \Big]

Note, that individual elements in the vectors
:math:`\mathbf{x}^{l-1}(\mathbf{x})` and :math:`\mathbf{x}^{l-1}(\mathbf{x}')`
are independent, and thus, all tuples
:math:`(x_j^{l-1}(\mathbf{x}), x_j^{l-1}(\mathbf{x}')` are drawn independently
from the bivariate Gaussian with a covariance matrix prescribed by the kernel
induced in the previous layer.

This scheme allows us to estimate the kernel of a multi-layer RBF network
recursively, using the analytic expression
:math:`V_G(\mathbf{x}, \mathbf{x}', d)` combined with MC samples from the GP
induced by the previous layer.
"""
import math
import torch
from torch.distributions import Normal, MultivariateNormal
import traceback
from warnings import warn

class RBFNetKernel:
    r"""A kernel function estimator for an infinite-width RBF networks.

    This class estimates the kernel of networks as implemented in
    :class:`finite_width.rbf_net.StackedRBFNet` when
    considering the infinite-width limit :math:`N_l \rightarrow \infty`.

    Args:
        sigma2_u (float): The variance of centers :math:`\sigma^2_u`.
        sigma2_w (float): The variance of weights :math:`\sigma^2_w`.
        sigma2_b (float): The variance of biases :math:`\sigma^2_b`.
        n_lin_hidden_units (list or tuple): Number :math:`M_l` of linear units
            :math:`\mathbf{x}^l` for each layer :math:`l`, except the last one
            as :math:`M_L = 1`.

            Note:
                :math:`N_l` is assumed to be infinite!
        bandwidth (float): The bandwidth parameter :math:`\sigma_g^2`.

    """
    def __init__(self, sigma2_u=1., sigma2_w=1., sigma2_b=1.,
                 n_lin_hidden_units=(), bandwidth=1.):
        self._n_layer = 1 + len(n_lin_hidden_units)
        self.sigma2_u = sigma2_u
        self._sigma2_w = sigma2_w
        self._sigma2_b = sigma2_b
        self._n_hidden = n_lin_hidden_units
        self._bandwidth = bandwidth

    def kernel_analytic(self, X):
        """Computing the kernel values analytically.

        Note:
            Currently, an analytic kernel is only for the 1-layer case (standard
            RBF net) available!

        Args:
            (....): See docstring of method :func:`nngp.mlp_kernel.init_kernel`.

        Returns:
            (torch.Tensor): The kernel matrix, whose shape depends on ``X``, see
            return value of function :func:`analytic_v_gaussian`.
        """
        if self._n_layer != 1:
            raise RuntimeError('Analytic kernel expression for ' + \
                '%d-layer RBF network not available.' % (self._n_layer))

        return self._sigma2_b + self._sigma2_w * \
            analytic_v_gaussian(X, sigma2_u=self.sigma2_u,
                                bandwidth=self._bandwidth)

    def kernel_mc(self, X, num_samples=100):
        """Kernel estimation via Monte-Carlo sampling.

        This method estimates the kernel values by recursively approximating
        Eq :eq:`rbf-net-kernel-recursion` using Monte-Carlo sampling.

        Args:
            (....): See docstring of method :meth:`kernel_analytic`.
            num_samples (int): The number of MC samples used for every expected
                value that needs to be estimated.

        Returns:
            (torch.Tensor): See docstring of method :meth:`kernel_analytic`.
        """
        K = self._sigma2_b + self._sigma2_w * \
                analytic_v_gaussian(X, sigma2_u=self.sigma2_u,
                                    bandwidth=self._bandwidth)

        if self._n_layer == 1:
            return K

        # We distinguish two cases: If a tensor is given, then the whole kernel
        # matrix needs to be approximated (also, for each step of the recursion,
        # the whole matrix is needed). If a tuple of tensors is given, then we
        # only need to compute the 2x2 kernel matrix for each pair in order to
        # solve the recursion.
        if not isinstance(X, torch.Tensor):
            X1 = X[0]
            X2 = X[1]
            if X1.ndim == 1:
                X1 = X1.reshape(-1, 1)
                X2 = X2.reshape(-1, 1)

            # We need to build a batch of 2x2 matrices comprising for the i-th
            # tuple `(x_i1, x_i2) `the kernel values `K(x_i1, x_i1)`,
            # `K(x_i2, x_i2)` and `K(x_i1, x_i2)`
            K_ij = K
            K_ii = self._sigma2_b + self._sigma2_w * \
                analytic_v_gaussian((X1, X1), sigma2_u=self.sigma2_u,
                                    bandwidth=self._bandwidth)
            K_jj = self._sigma2_b + self._sigma2_w * \
                analytic_v_gaussian((X2, X2), sigma2_u=self.sigma2_u,
                                    bandwidth=self._bandwidth)

            K = K_ij.view(-1, 1, 1).repeat(1, 2, 2)
            K[:, 0, 0] = K_ii
            K[:, 1, 1] = K_jj

            raise NotImplementedError()
            #if comp_type == 'mc' and kwargs['parallel_comp']:
            #    K = self._kernel_mc_recursion_parallel(K, 1, self._n_layer,
            #        kwargs['num_samples'])
            #    K = K[:, 0, 1]
            #    return K

        else:
            raise NotImplementedError()
            #return self._kernel_mc_recursion_parallel(K_0, 1,
            #    self._n_layer, num_samples)

def analytic_v_gaussian(X, sigma2_u=1., bandwidth=1.):
    r"""Analytically compute Eq. :eq:`v_g_williams`.

    This functions analytically computes the expression

    .. math::

        V_G(\mathbf{x}, \mathbf{x}') = \mathbb{E}_{\mathbf{u}} \
            \big[ h(\mathbf{x}, \mathbf{u}) h(\mathbf{x}', \mathbf{u}) \big]

    Args:
        (....): See docstring of class :class:`RBFNetKernel` and function
            :func:`nngp.mlp_kernel.init_kernel`.

    Returns:
        (torch.Tensor): If ``X`` is passed as a single tensor, the symmetric
        matrix (2D tensor) with entries
        :math:`V_G(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})` is returned.
        Otherwise, a vector is returned.
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
    N = X1.shape[0]
    d = X1.shape[1]

    # Compute all the required factors.
    o_e = math.sqrt(1 / (2 / bandwidth + 1 / sigma2_u))
    o_u = math.sqrt(sigma2_u)
    o2_s = 2 * bandwidth + bandwidth**2 / sigma2_u
    o2_m = 2 * sigma2_u + bandwidth

    r = (o_e / o_u)**d

    x1x1_term = torch.exp(- torch.sum(X1 * X1, dim=1) / (2 * o2_m))

    if X2 is None:
        x2x2_term = x1x1_term.clone()

        x1x1_term = x1x1_term.view(-1, 1).repeat(1, N)
        x2x2_term = x2x2_term.view(1, -1).repeat(N, 1)
    else:
        x2x2_term = torch.exp(- torch.sum(X2 * X2, dim=1) / (2 * o2_m))

    if X2 is None:
        diff = X1.view(N, 1, d) - X1.view(1, N, d)
        x1x2_term = (diff * diff).sum(dim=2)
    else:
        diff = X1 - X2
        x1x2_term = (diff * diff).sum(dim=1)
    x1x2_term = torch.exp(- x1x2_term / (2 * o2_s))

    return r * x1x1_term * x1x2_term * x2x2_term

def mc_v_gaussian(X, sigma2_u=1., bandwidth=1., num_samples=1000):
    """Estimate Eq. :eq:`v_g_williams` via Monte-Carlo.

    Args:
        (....): See docstring of function :func:`analytic_v_gaussian`.
        num_samples (int): Number of MC samples.

    Returns:
        (torch.Tensor): See docstring of function :func:`analytic_v_gaussian`.
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
    #N = X1.shape[0]
    d = X1.shape[1]

    # MC samples.
    u_std = math.sqrt(sigma2_u)
    u_dist = Normal(0, torch.tensor(u_std).to(X1.device))

    u = u_dist.sample((num_samples, d))

    h_x1 = torch.exp(- 1. / (2. * bandwidth) * \
        ((X1[:, None, :] - u[None, :, :])**2).sum(dim=2))
    if X2 is not None:
        h_x2 = torch.exp(- 1. / (2. * bandwidth) * \
            ((X2[:, None, :] - u[None, :, :])**2).sum(dim=2))

    if X2 is None:
        V = 1 / num_samples * (h_x1 @ h_x1.T)
    else:
        V = (h_x1 * h_x2).mean(dim=1)

    return V

if __name__ == '__main__':
    pass


