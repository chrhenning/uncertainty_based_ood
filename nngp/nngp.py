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
# @title          :nngp/nngp.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/19/2021
# @version        :1.0
# @python_version :3.8.5
r"""
Deep Neural Network as Gaussian Process
---------------------------------------

The module :mod:`nngp.nngp` implements helper functions for Bayesian inference
with Gaussian Processes with a focus on kernels derived from neural network
architectures when taken to the infinite-width limit
(cf. :mod:`nngp.mlp_kernel`).

Specifically, we consider a Gaussian Process
:math:`\mathcal{GP}\big(\mu(x), k(x, x')\big)` with mean function
:math:`\mu(\cdot)` and kernel :math:`k(\cdot, \cdot)`. Unless specified
otherwise, we assume the mean function to be :math:`\mu(x) = 0`. Note, that any
multivariate Gaussian prescribed by the :math:`\mathcal{GP}` at a given set of
input locations is consistent (marginalization from any superset of locations
will always lead to the same distribution) and adheres exchangibility (order of
input locations doesn't affect the distribution except for repositioning the
corresponding function values).

For any given set of inputs :math:`X = x_1, \dots, x_n`, the
:math:`\mathcal{GP}` allows us to specify a prior distribution over function
values :math:`p(f_1, \dots, f_n; x_1, \dots, x_n) \equiv p(F; X)`.

In addition to inputs :math:`x` and function values :math:`f`, we consider
observations :math:`y`, which are obtained via a likelihood function
:math:`p(y \mid f)`.

Using the prior distribution over functions (the :math:`\mathcal{GP}`) and a
dataset :math:`\mathcal{D} = \{(x_n, y_n)\}_{n=1}^N` with inputs :math:`X` and
targets :math:`Y`, one can form a posterior distribution over function values
:math:`f` at an unknown location :math:`x^*` via

.. math::

    p(f \mid \mathcal{D}; x^*) = p(f \mid Y; x^* X) = \frac{1}{p(Y; X)} \
    \int p(Y \mid F) p(F, f; X, x^*) \, dF

Please see
`Rasmussen and Williams <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`__
for a broader introduction into Gaussian Processes.
"""
import torch
from warnings import warn

def inference_with_isotropic_gaussian_ll(Y, K_train, K_test, K_all, var=1e-10,
                                         L_mat=None, return_cov=False):
    r"""Bayesian inference with Gaussian likelihood and :math:`\mathcal{GP}`
    prior.

    Here, we consider the case
    :math:`p(Y \mid F) = \mathcal{N}(Y; F, \sigma_\epsilon^2 I)`, where the
    posterior predictive :math:`p(f \mid \mathcal{D}; x^*)` can be analytically
    computed

    .. math::

        p(f \mid \mathcal{D}; x^*) &=  \mathcal{N}(f; \mu^*, \Sigma^*) \\ \
        \mu^* &= K(x^*, X) \big( K(X, X) + \sigma_\epsilon^2 I \big)^{-1} Y \\ \
        \Sigma^* &= k(x^*, x^*) - K(x^*, X) \big( K(X, X) + \
            \sigma_\epsilon^2 I \big)^{-1} K(X, x^*)

    Args:
        Y (torch.Tensor): The labels :math:`Y` from the training set encoded as
            vector of shape ``[m]`` or ``[m, 1]``.
        K_train (torch.Tensor): The training data kernel matrix :math:`K(X, X)`.
        K_test (torch.Tensor): The test data kernel values :math:`k(x^*, x^*)`.
            This is a vector either of shape ``[n]``, where ``n`` is the number
            test points, or of shape ``[n, 1]``.
        K_all (torch.Tensor): The kernel values between train and test points
            :math:`K(x^*, X)`. This is expected to be matrix of shape ``[n,m]``,
            where ``m`` is the number of training and ``n`` the number of test
            points, or simply a vector of shape ``[m]``, if there is only one
            test point.
        var (float): The variance :math:`\sigma_\epsilon^2` of the likelihood.
        L_mat (torch.Tensor, optional): The matrix :math:`L` resulting from a
            Cholesky decomposition of :math:`K(X, X) + \sigma_\epsilon^2 I`.
            If provided, the arguments ``K_train`` and ``var`` are ignored.

            The function :func:`cholesky_adaptive_noise` may be helpful to
            compute ``L_mat``.
        return_cov (bool): If ``True``, the return value ``cov`` will be the
            full covariance matrix. However, this option requires ``K_test``
            to be the full ``[n, n]`` kernel matrix.

    Returns:
        (tuple): Tuple containing:

        - **mean** (torch.Tensor): A tensor of shape ``[n]``, where ``n`` is the
          number of test points. The tensor encodes the mean for each test point
          of the posterior predictive :math:`\mu^*`.
        - **cov** (torch.Tensor): Same as ``mean`` but encoding the variance
          :math:`\Sigma^*` of each test point, i.e., the diagonal of the full
          covariance matrix.
    """
    m = K_train.shape[0] if L_mat is None else L_mat.shape[0]
    n = K_test.shape[0]
    assert Y.numel() == m
    assert K_all.numel() == m*n

    if Y.ndim == 1:
        Y = Y.view(-1, 1)
    if return_cov:
        assert K_test.numel() == n*n and K_test.ndim == 2
    elif K_test.ndim == 2:
        K_test = K_test.view(-1)
    if K_all.ndim == 1:
        assert n == 1
        K_all = K_all.view(n, m)

    #inv_K = torch.linalg.inv(K_train + var * torch.eye(m).to(K_train.device))
    #mu = torch.matmul(K_all, torch.matmul(inv_K, Y))
    #if return_cov:
    #    sigma = K_test - torch.matmul(K_all, torch.matmul(inv_K, K_all.T))
    #else:
    #    #sigma = K_test - torch.bmm(K_all.view(n, 1, m), torch.matmul(inv_K,
    #    #    K_all.view(n, m, 1))).squeeze()
    #    sigma = K_test - (K_all * torch.matmul(inv_K,
    #        K_all.view(n, m, 1)).squeeze(dim=2)).sum(dim=1)

    # Note, direct matrix inversion is considered extremely numerically
    # unstable. Therefore, Rasmussen et al. propose the use of Cholesky
    # decomposition, see Appendix A.4 in
    # http://www.gaussianprocess.org/gpml/chapters/RW.pdf
    if L_mat is None:
        L = torch.linalg.cholesky(K_train + \
                                  var * torch.eye(m).to(K_train.device))
    else:
        L = L_mat
    alpha = torch.triangular_solve(torch.triangular_solve(Y, L, upper=False)[0],
                                   L, upper=False, transpose=True)[0]
    mu = torch.matmul(K_all, alpha)

    v = torch.triangular_solve(K_all.T, L, upper=False)[0]
    if return_cov:
        sigma = K_test - torch.matmul(v.T, v)
    else:
        sigma = K_test - (v * v).sum(dim=0)

    if torch.any(sigma < 0):
        sigma[sigma < 0] = 1e-5
        warn('Some entries of the covariance matrix are negative and set to ' +
             '1e-5!')

    return mu.squeeze(), sigma

def gen_inference_kernels(X_train, X_test, kernel_func, compute_K_train=True,
                          full_K_test=False):
    r"""Generate the kernel matrices required for inference.

    This function generates the kernel matrices / vectors :math:`K(X, X)`,
    :math:`K(x^*, X)` and :math:`K(x^*, x^*)`, where :math:`X` are training
    inputs and :math:`x^*` are unseen points.

    Thus, the function can be seen as helper function for functions like
    :func:`inference_with_isotropic_gaussian_ll`.

    Args:
        X_train (torch.Tensor): A batch of ``m`` training inputs. The tensor
            should have shape ``[m, d_in]``, where ``d_in`` is the input
            dimensionality. For scalar inputs, one may also pass a tensor of
            shape ``[m]``.
        X_test (torch.Tensor):A batch of ``n`` unseen test inputs.
        kernel_func (func): The kernel function :math:`k(x, x')`. It is expected
            to have an interface for a single input ``X`` as described in
            the docstring of function:`nngp.mlp_kernel.init_kernel`.

            .. code-block:: python

                def kernel_func(X):
                    # Compute kernel values.
                    return K

        compute_K_train (bool): Whether the kernel matrix :math:`K(X, X)`
            should be computed. If ``False``, the return value ``K_train`` is
            ``None``.
        full_K_test (bool): Whether the full kernel matrix :math:`K(x^*, x^*)`
            of shape ``[n, n]`` should be computed.

    Returns:
        (tuple): Tuple containing:

          - **K_train** (torch.Tensor or None): :math:`K(X, X)`, a tensor of
            shape ``[m, m]``.
          - **K_test** (torch.Tensor): :math:`K(x^*, x^*)`, a tensor of shape
            ``[n]``
          - **K_all** (torch.Tensor): :math:`K(x^*, X)`, a tensor of shape
            ``[n,m]``
    """
    if compute_K_train:
        K_train = kernel_func(X_train)
    else:
        K_train = None

    if full_K_test:
        K_test = kernel_func(X_test)
    else:
        K_test = kernel_func((X_test, X_test))

    # Contruct tuples between all train samples and all test samples.
    if X_train.ndim == 1: # `d_in == 1`
        X_train = X_train.view(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.view(-1, 1)

    m = X_train.shape[0]
    n = X_test.shape[0]

    X_all = (X_train.repeat(n, 1),
             X_test.view(n, 1, -1).repeat(1, m, 1).view(n*m, -1))
    K_all = kernel_func(X_all)

    K_all = K_all.view(n, m)

    return K_train, K_test, K_all

def cholesky_adaptive_noise(K_train, var=1e-10, var_step=2.):
    r"""Cholesky decomposition of a kernel matrix with noise perturbation.

    This function computes the Cholesky decomposition of:

    .. math::

        L L^T = K(X, X) + \sigma_\epsilon^2 I

    As kernel matrices :math:`K(X, X)` may easily be (numerically) singular,
    tuning the noise :math:`\sigma_\epsilon^2` is crucial. Therefore, this
    method will iteratively increase the noise level until the matrix becomes
    non-singular.

    Args:
        (....): See docstring of method :meth:`kernel_efficient`.
        var (float or list): The initial variance :math:`\sigma_\epsilon^2`.
            If a list of values is provided, then each value in this list is
            consecutively tested until a non-singular matrix is constructed.
            Note, we assume that the list is sorted from small to large. If none
            of the elements in this list will lead to a non-singular matrix, an
            exception is raised.
        var_step (float): If ``var`` is a single value, then the value specified
            here will be iteratively multiplied to increase the variance
            :math:`\sigma_\epsilon^2` (therefore ``var_step > 1`` is required).

    Returns:
        (tuple): Tuple containing:
          - **L** (torch.Tensor): The matrix :math:`L` resulting from the
            successful Cholesky decomposition.
          - **var_chosen** (float): The variance :math:`\sigma_\epsilon^2` that
            was chosen to obtain ``L``.
    """
    m = K_train.shape[0]

    if not isinstance(var, (list, tuple)):
        assert var_step > 1.

    i = 0
    while True:
        if isinstance(var, (list, tuple)):
            if i >= len(var):
                raise RuntimeError('List of variances didn\'t contain high ' +
                                   'enough values.')
            curr_var = var[i]
        else:
            if i == 0:
                curr_var = var
            else:
                curr_var *= var_step

        try:
            L = torch.linalg.cholesky(K_train + curr_var * torch.eye(m).to( \
                K_train.device))
        except:
            i += 1
            continue

        return L, curr_var

if __name__ == '__main__':
    pass


