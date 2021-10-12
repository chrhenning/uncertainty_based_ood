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
# @title          :finite_width/ridgelet_prior.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/30/2021
# @version        :1.0
# @python_version :3.8.10
r"""
The Ridgelet Prior
------------------

The module :mod:`finite_width.ridgelet_prior` provides an implementation of the
Ridgelet prior as introduced by

    `Matsubara et al. <https://arxiv.org/abs/2010.08488>`__,
    The Ridgelet Prior: A Covariance Function Approach to Prior Specification
    for Bayesian Neural Networks, 2021.

The ridgelet prior allows to specify a weight prior of a BNN (with given
architecture) to approximately capture the function space prior as described by
a target Gaussian process (GP).

The paper focuses on MLP networks. Note, that only certain activation functions
are supported (e.g., compare Table 1 in the paper).

In practice, the computation of the prior is rather expensive and currently
only interesting for toy examples. In addition, the GP prior can only be
approximated on a compact domain :math:`\mathcal{X}`, and the prior behavior
outside this domain would need to be investigated.

In the following, we comment a bit on the construction of this prior to gain a
better intuition.

The prior construction is based on the Ridgelet transform, which has the
following property for a suitably chosen pair of :math:`\phi(\cdot)` and
:math:`\psi(\cdot)`:

.. math::

    f(\mathbf{x}) = \int_{\mathbb{R}^{d+1}} \bigg( \int_{\mathbb{R}^d} \
        \psi(\mathbf{w}^T \mathbf{x} + b) f(\mathbf{x}) d\mathbf{x} \bigg) \
        \phi(\mathbf{w}^T \mathbf{x} + b) d\mathbf{w} db

where :math:`d` is the dimensionality of the inputs. As a next step, those
integrals are discretized, which makes the connection to a single hidden layer
neural network obvious:

.. math::
    :label: ridgelet-discretization

    I_{\sigma, D, N} [f](\mathbf{x}) = \sum_{i=1}^N v_i \bigg( \
        \sum_{j=1}^D u_j f(\mathbf{x}_j) \psi(\mathbf{w}_i^T \mathbf{x}_j \
        + b_i) \bigg) \phi(\mathbf{w}_i^T \mathbf{x} + b_i)

where :math:`D` is the number of points :math:`\mathbf{x}_j` chosen to perform
numerical integration on the compact domain :math:`\mathcal{X}`, and
:math:`v_i`, :math:`u_j` are the cubature points corresponding to the two
integrals.

The discretisation of the inner-integral is described in *Assumption 2* in the
paper. In our case, the compact domain :math:`\mathcal{X}` is simply defined by
the hyper-cube :math:`[S, S]^d` on which :math:`\{\mathbf{x}_j\}_{j=1}^D` form
a regular grid with cubature weights :math:`u_j = (2S)^d/D`.

The discretisation of the outer integral is described in *Assumption 3* in the
paper and based on a change of measure such that a Monte-Carlo integration
is feasible. Therefore, a set :math:`\{(\mathbf{w}_i, b_i)\}_{i=1}^N` is drawn
from the first layer's weight prior :math:`\mathcal{N}(0, \sigma_w^2)` and
:math:`\mathcal{N}(0, \sigma_b^2)`, and the cubature weights are set to
:math:`v_i = Z/N` with :math:`Z = (2\pi)^\frac{1}{2} \sigma_w^d \sigma_b`.

In the limit :math:`\sigma, N, D \rightarrow \infty` the estimator
:math:`I_{\sigma, D, N} [f](x)` should converge to the target :math:`f(x)`.

Eq :eq:`ridgelet-discretization` is reminiscent of a single hidden layer MLP
with 1 output neuron

.. math::

    \text{NN}(\mathbf{x}) = \sum_{i=1}^N w_i^1 \
        \phi( (\mathbf{w}_i^0)^T \mathbf{x} + b_i^0)

with outputs weights

.. math::

    w_i^1 = v_i \sum_{j=1}^D u_j f(\mathbf{x}_j) \
        \psi((\mathbf{w}_i^0)^T \mathbf{x}_j + b_i^0)

Note, the dependence of the output weights on the input weights, which is an
important construction detail of this prior.

Assume a matrix :math:`\Psi^0` with entries
:math:`[\Psi^0]_{i,j} = v_i u_j \psi((\mathbf{w}_i^0)^T \mathbf{x}_j + b_i^0)`
and a GP :math:`f \sim \mathcal{GP}(\mathbf{m}, K)`. In this case, we
can express the output weights as :math:`\mathbf{w}^1 = \Psi^0 \mathbf{f}` with
:math:`\mathbf{f}_j = f(\mathbf{x}_j)`. Since :math:`\mathbf{f}_j` is a Gaussian
random variable, also :math:`\mathbf{w}^1` is a Gaussian random variable with
mean :math:`\mathbb{E}[\mathbf{w}^1] = \Psi^0 \mathbf{m}` and covariance matrix

.. math::

    \mathbb{E}\big[(\mathbf{w}^1 - \mathbb{E}[\mathbf{w}^1]) \
        (\mathbf{w}^1 - \mathbb{E}[\mathbf{w}^1])^T\big] = \
        \Psi^0 \mathbb{E}\big[(\mathbf{f} - \mathbf{m}) \
        (\mathbf{f} - \mathbf{m})^T\big] (\Psi^0)^T = \Psi^0 K (\Psi^0)^T

In definition 1, the authors propose the following prior construction for a
multi-layer MLP with :math:`L` layers: biases are drawn from
:math:`b_i^{l-1} \sim \mathcal{N}(0, \sigma_b^2)`, first layer weights are
drawn from :math:`\mathbf{w}_i^0 \sim \mathcal{N}(0, \sigma_w^2 I)` and the
weights of all remaining layers are drawn according to

.. math::

    \mathbf{w}^l_i \mid \{(\mathbf{w}_r^{l-1}, b_r^{l-1}) : r = 1,\dots,N_l \} \
        \sim \mathcal{N}(\Psi^{l-1} \mathbf{m}, \Psi^{l-1} K (\Psi^{l-1})^T)

where

.. math::

    [\Psi^{l-1}]_{i,j} = v_i u_j \psi\big( (\mathbf{w}_i^{l-1})^T \
        \phi\big(\mathbf{z}^{l-1} (\mathbf{x}_j)\big) + b_i^{l-1} \big)

Example:
    .. code-block:: python

        from hypnettorch.mnets import MLP
        from sklearn.gaussian_process.kernels import RBF

        d = 1
        S = 5

        X = rp.regular_grid(cube_size=S, res_per_dim=8, grid_dim=d)

        rbf_kernel = RBF(length_scale=1.)

        m = torch.zeros(X.shape[0])
        K = torch.from_numpy(rbf_kernel(X)).type(torch.FloatTensor)

        net = MLP(n_in=d, n_out=1, hidden_layers=(10,))

        sample, dists = rp.ridgelet_prior_sample(net, phi_name='relu',
            cube_size=S, grid_points=X, gp_mean=m, gp_cov=K, return_dist=True)

        print('log-prob: ', rp.ridgelet_prior_logprob(sample, dists=dists))

        inputs = torch.rand(4, d)
        preds = net.forward(inputs, weights=sample)
"""
from hypnettorch.mnets import MLP
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
from warnings import warn

def ridgelet_prior_sample(net=None, hidden_widths=None, phi_name='relu',
                          cube_size=5, grid_points=None, gp_mean=None,
                          gp_cov=None, sigma2_w=1., sigma2_b=1.,
                          return_dist=False, sample=None):
    """Generate a sample from the Ridgelet prior.

    This function will generate a sample from the Ridgelet prior according to
    the construction outlined above (see :mod:`finite_width.ridgelet_prior`). It
    thus only applies to plain MLP networks. Therefore, one may either specify
    an instance of class :class:`hypnettorch.mnets.mlp.MLP` via option ``net``,
    or specify a list of hidden layer weights via option ``hidden_widths``. The
    basic construction assumes bias terms at every hidden layer, but not at the
    output layer. For simplicity, this assumption is carried on in this
    implementation. Therefore, if ``net`` has output bias weights, these are
    set to zero in every sample. On the other hand, if ``net`` has no bias terms
    in hidden layers, an error is raised.

    Args:
        net (hypnettorch.mnets.mlp.MLP): A plain MLP instance. The returned
            weight sample will have shapes according to attribute
            ``param_shapes``. Note, that this function assumes that attribute
            ``has_bias`` evaluates to ``True``.
        hidden_widths (list or tuple, optional): If ``net`` is not provided,
            then a list of hidden widths :math:`N_l` has to be provided. The
            output shape of the network is assumed to be 1.
        phi_name (str): The activation function used in the network. The
            following options are supported:

              - ``'relu'``: :math:`\phi(z) = \max(0, z)`

        cube_size (float): The discretisation of the inner-integral is performed
            using a regular grid on the hyper-cube :math:`[S, S]^d`. Argument
            ``cube_size`` refers to :math:`S`, while argument ``grid_points``
            contains the actual grid points.
        grid_points (torch.Tensor): A tensor of shape ``[D, d]`` containing the
            grid points :math:`\mathbf{x}_j`.
        gp_mean (torch.Tensor): A tensor of shape ``[D]`` containing the target
            GP mean :math:`\mathbf{m}` at all the ``grid_points``.
        gp_cov (torch.Tensor): A tensor of shape ``[D, D]`` containing the
            target GP covariance matrix :math:`K`, i.e., the kernel value at all
            ``grid_points`` :math:`k(\mathbf{x}_i, \mathbf{x}_j)`.
        sigma2_w (float): The variance :math:`\sigma_w^2` of the first layer
            weights.
        sigma2_b (float): The variance :math:`\sigma_b^2` of all biases.
        return_dist (bool): If ``True``, a list of distribution objects from
            class :class:`torch.distributions.normal.Normal` or
            :class:`torch.distributions.multivariate_normal.MultivariateNormal`
            is returned in addition to the weight sample. This list can be used
            to compute the log-probability of the sample by summing the
            individual log-probabilities.

            This list will have the same order as the returned list of weight
            samples.

            Note:
                Those distribution instances are only valid to compute the
                log-probability for the returned sample, not for samples in
                general, as every new sample recursively defines a different
                list of distributions!
        sample (list or tuple): A list of torch tensors. If provided, no new
            sample will be generated, instead the passed ``sample`` will be
            returned by this function. Therefore, this option is only meaningful
            in combination with option ``return_dist`` in case the
            log-probability of a given sample should be computed.

    Returns:
        (list): A list of tensors.
    """
    # TODO Ideally we add an argument like this:
    #    rand_state (torch.Generator, optional): A generator to make random
    #        sampling reproducible.
    # But that would require us (potentially) to implement our own `rsample`
    # using the build-in sampling functions of PyTorch/

    assert net is not None or hidden_widths is not None
    #assert grid_points is not None and gp_mean is not None and \
    #    gp_cov is not None

    if net is not None and hidden_widths is not None:
        raise ValueError('You may either specify "net" or "hidden_widths", ' +
                         'not both!')

    # Determine the `hidden_widths` from the network.
    d_out = 1
    if net is not None:
        assert isinstance(net, MLP)
        # Sanity check. This might fail, if the user uses a custom ReLU
        # implementation.
        if hasattr(net, '_a_fun') and phi_name == 'relu':
            assert isinstance(net._a_fun, torch.nn.ReLU)
        elif hasattr(net, '_a_fun') and phi_name == 'tanh':
            assert isinstance(net._a_fun, torch.nn.Tanh)

        num_layers = 0
        for meta in net.param_shapes_meta:
            if meta['name'] not in ['bias', 'weight']:
                raise RuntimeError('Provided network is not plain MLP!')
            if meta['name'] == 'weight':
                num_layers += 1
        num_layers -= 1 # we are interested in num hidden layers.

        hidden_widths = [-1] * num_layers

        for i, meta in enumerate(net.param_shapes_meta):
            if meta['name'] == 'weight':
                s = net.param_shapes[i]
                if  meta['layer'] < num_layers:
                    hidden_widths[meta['layer']] = s[0]
                else:
                    d_out = s[0]

        if sample is not None:
            assert len(sample) == len(net.param_shapes)
            weights = [None] * (num_layers+1)
            biases = [None] * (num_layers+1)

            for i, meta in enumerate(net.param_shapes_meta):
                if meta['name'] == 'weight':
                    weights[meta['layer']] = sample[i]
                else:
                    biases[meta['layer']] = sample[i]

    if len(grid_points.shape) == 1:
        grid_points.reshape(-1, 1)
    assert len(grid_points.shape) == 2
    D = grid_points.shape[0]
    d = grid_points.shape[1]

    device = grid_points.device

    sigma_w = np.sqrt(sigma2_w)
    sigma_b = np.sqrt(sigma2_b)

    weight_dists = []
    bias_dists = []
    if sample is None:
        weights = []
        biases = []
    else:
        if net is None:
            # TODO Decompose `sample` into list of `weights` and `biases`.
            raise NotImplementedError()

    # Compute choesky of Kernel matrix.
    # Compute: K = L L^T, such that C = (P^T L) (P^T L)^T
    # Note, weights can be samples as w = m + (P^T L) eps,
    # where `eps` is white noise. However, this scheme might due to numerical
    # issues cause C = (P^T L) (P^T L)^T to be not invertable. Thus, if
    # distribution objects need to be created, this simple scheme is not
    # sufficient as invertibility of `C` needs to be ensured.
    gp_cov_L = None
    if not return_dist:
        gp_cov_L = _cholesky_noise_adaptive(gp_cov)

    # Sample first layer weights.
    weight_dists.append(Normal(torch.zeros([hidden_widths[0], d]).to(device),
                               sigma_w))
    bias_dists.append(Normal(torch.zeros([hidden_widths[0]]).to(device),
                             sigma_b))

    if sample is None:
        weights.append(weight_dists[-1].rsample())
        biases.append(bias_dists[-1].rsample())

    if len(hidden_widths) == 0:
        raise ValueError('The Ridgelet prior construction requires at least ' +
                         'one hidden layer.')
    if len(hidden_widths) > 1:
        raise NotImplementedError('Ridgelet prior not implemented for more ' +
                                  'than one hidden layer yet.')

    # Cubature weights
    # TODO Incorporate mollifier to compute `x_j` dependent weight.
    u_j = (2 * cube_size)**d / D
    # Note, those cubature weights ideally depend on the determinant of the
    # previous layer's covariance matrix.
    v_i_unnorm = (2*np.pi)**(1/2) * sigma_w**d * sigma_b

    # Sample next layer weights conditioned on previous layer weights.
    z = grid_points
    for l in range(len(hidden_widths)):
        last = l == len(hidden_widths) - 1
        n_in = hidden_widths[l]
        n_out = d_out if last else hidden_widths[l+1]

        if last:# No bias weights in the output layer.
            bias_dists.append(None)
            if sample is None:
                biases.append(torch.zeros([n_out]).to(device))
        else:
            bias_dists.append(Normal(torch.zeros([n_out]).to(device), sigma_b))
            if sample is None:
                biases.append(bias_dists[-1].rsample())

        W_prev = weights[l]
        b_prev = biases[l]

        if l > 0:
            if phi_name == 'relu':
                z = torch.nn.functional.relu(z)

        z = z @ W_prev.T + b_prev[None, :]

        if phi_name == 'relu' and d in [1, 2]: # Much faster.
            psi = _ass_func_relu_manual(z, d)
        elif phi_name == 'tanh' and d == 1: # Much faster.
            psi = _ass_func_tanh_manual(z, d)
        else:
            psi = _ass_func(z, d, phi_name=phi_name)

        v_i = v_i_unnorm / n_in
        psi *= (u_j * v_i)

        # Mean of weight distribution.
        gauss_m = psi.T @ gp_mean

        # Covariance of weight distribution.
        if return_dist: # Sample and create distribution object.
            gauss_C = psi.T @ gp_cov @ psi
            gauss_L = _cholesky_noise_adaptive(gauss_C)

            weight_dists.append(MultivariateNormal(gauss_m, scale_tril=gauss_L))
        else: # Only sample.
            gauss_L = psi.T @ gp_cov_L
            weight_dists.append(None)

        if sample is None:
            W = [] # We need to draw a weight vector for every output neuron.
            for _ in range(n_out):
                if return_dist:
                    W.append(weight_dists[-1].rsample())
                else:
                    eps = torch.normal(torch.zeros((D)).to(device))
                    W.append(gauss_m + gauss_L @ eps)
            weights.append(torch.stack(W))

    # Merge weights and biases in single list.
    if net is None:
        dists = []
        sample = []

        for i, wdist in enumerate(weight_dists):
            last = i == len(weight_dists) - 1
            dists.append(wdist)
            sample.append(weights[i])
            if not last:
                dists.append(bias_dists[i])
                sample.append(biases[i])
    else:
        dists = [None] * len(net.param_shapes)
        sample = [None] * len(net.param_shapes)

        for i, meta in enumerate(net.param_shapes_meta):
            l = meta['layer']
            if meta['name'] == 'weight':
                dists[i] = weight_dists[l]
                sample[i] = weights[l]
            else:
                dists[i] = bias_dists[l]
                sample[i] = biases[l]

    if return_dist:
        return sample, dists
    return sample

def ridgelet_prior_logprob(sample, dists=None, net=None, hidden_widths=None,
                           phi_name='relu', cube_size=5, grid_points=None,
                           gp_mean=None, gp_cov=None, sigma2_w=1., sigma2_b=1.):
    """Compute the log-probability of a sample from the Ridgelet prior.

    This function computes the logprob of a sample returned by function
    :func:`ridgelet_prior_sample`.

    Args:
        (....): See docstring of function :func:`ridgelet_prior_sample`.
        dists (list): If option ``return_dist`` was used to obtain the
            ``sample`` via function :func:`ridgelet_prior_sample`, then the
            returned distributions corresponding to the given ``sample`` can be
            passed here.

            If ``dists`` is provided, then all other options (except ``sample``)
            don't need to be specified.

            Note:
                Every sample has its own set of distributions generated by
                function :func:`ridgelet_prior_sample`!

    Returns:
        (float): The log-probability :math:`\log p(W)` for the given sample
        :math:`W`.
    """
    if dists is None:
        _, dists = ridgelet_prior_sample(net=net, hidden_widths=hidden_widths,
            phi_name=phi_name, cube_size=cube_size, grid_points=grid_points,
            gp_mean=gp_mean, gp_cov=gp_cov, sigma2_w=sigma2_w,
            sigma2_b=sigma2_b, return_dist=True, sample=sample)

    logprob = 0
    for i, d in enumerate(dists):
        if isinstance(d, Normal):
            logprob += d.log_prob(sample[i]).sum()
        elif isinstance(d, MultivariateNormal):
            # One logprob per neuron (row in sample matrix) is computed.
            logprob += d.log_prob(sample[i]).sum()
        else:
            assert d is None # Bias of output layer.

    return logprob

def regular_grid(cube_size=5, res_per_dim=100, grid_dim=1, device=None):
    """Create a regular grid.

    This function creates a grid containing points :math:`\mathbf{x}_j` used
    for numerical integraion in function :func:`ridgelet_prior_sample`.
    Specifically, the return value of this function can be passed as argument
    ``grid_points`` to :func:`ridgelet_prior_sample`.

    Args:
        (....): See docstring of function :func:`ridgelet_prior_sample`.
        res_per_dim (int): Resolution per dimension. In total, the grid will
            have ``res_per_dim**grid_dim`` points.
        grid_dim (int): The dimensionality :math:`d` of the grid.
        device (optional): The PyTorch device of the return value.

    Returns:
        (torch.Tensor): A tensor of shape ``[D, grid_dim]``, where ``D`` is the
        number of grid points.
    """
    x_grid = torch.linspace(-cube_size, cube_size, res_per_dim)

    mg = torch.meshgrid([x_grid] * grid_dim)

    X = torch.vstack(list(map(torch.ravel, mg))).T

    if device is not None:
        X = X.to(device)

    return X

def _cholesky_noise_adaptive(A):
    """Compute cholesky decomposition of :math:`A`.

    Diagonal elements of :math:`A` are perturbed until the matrix becomes
    positive definite.
    """
    try: # TODO move outside of loop.
        L = torch.linalg.cholesky(A)
    except:
        evs = torch.linalg.eigvalsh(A, UPLO='L')
        noise = 1e-5
        fail = True
        while fail:
            try:
                L = torch.linalg.cholesky(A + \
                    (-evs.min() + noise) * torch.eye(A.shape[0]).to(A.device))
                fail = False
            except:
                if noise > 1:
                    raise RuntimeError('Failed to make kernel matrix ' +
                                       'invertible.')
                noise *= 2
        warn('Kenel matrix inversion required diagonal noise adaptation ' +
             'of scale %f.' % (-evs.min() + noise))

    return L

def _ass_func(z, d, phi_name='relu'):
    """Compute the associated function :math:`\psi(z)`.

    This function computes the function :math:`\psi(z)` associated to an
    activation function :math:`\phi(z)`. See Table 1
    `here <https://arxiv.org/abs/2010.08488>`__ for examples.

    Args:
        z (torch.Tensor): Input :math:`\psi(z)`. The tensor ``z`` nay have
            arbitrary shape, but the function :math:`\psi(z)` is applied
            elementwise.
        d (int): Input dimensionality.

    Returns:
        (torch.Tensor): Returns the output of :math:`\psi(z)`.
    """
    if phi_name not in ['relu', 'tanh']:
        raise NotImplementedError()

    r = d % 2

    def _relu_base(z):
        return torch.exp(-z**2 / 2)

    def _relu_factor():
        return torch.Tensor([-2**(-d/2) * np.pi**(-(d-2*r+1) / 2)])

    def _tanh_base(z):
        return torch.exp(-z**2 / 2) * torch.sin(np.pi * z / 2)

    def _tanh_factor():
        return torch.Tensor([2**(-(d+r)/2) * np.pi**(-(d-r+2) / 2) * \
                             np.exp(np.pi**2 / 8)])

    if phi_name == 'relu':
        grad_order = d + r + 2
        base_func = _relu_base
        factor = _relu_factor()
    elif phi_name == 'tanh':
        grad_order = d - r + 2
        base_func = _tanh_base
        factor = _tanh_factor()

    ### Compute n-th derivative.
    # grad can only be applied to scalar inputs.
    orig_shape = z.shape
    z_flat = z.flatten()
    z_flat.requires_grad = True

    grads = torch.empty_like(z_flat)
    for i in range(z_flat.numel()):
        x = z_flat[i]
        y = base_func(x)
        for n in range(grad_order):
            last = n == grad_order - 1
            grad = torch.autograd.grad(y, x, grad_outputs=None,
                retain_graph=not last, create_graph=not last, only_inputs=True,
                allow_unused=False)
            y = grad[0]

        grads[i] = y

    grads = grads.reshape(orig_shape)

    return factor * grads

def _ass_func_relu_manual(z, d):
    """The associated function :math:`\psi` for a ReLU activation.

    This function uses manually computed derivatives and is not generally
    applicable.

    Args:
        (....): See docstring of function :func:`_ass_func`.

    Returns:
        (torch.Tensor): Returns the output of :math:`\psi(z)`.
    """
    r = d % 2

    def _base_func(z):
        return torch.exp(-z**2 / 2)

    def _factor():
        return torch.Tensor([-2**(-d/2) * np.pi**(-(d-2*r+1) / 2)])

    order = d + r + 2

    def _grad_1st(z):
        return -z * _base_func(z)

    def _grad_2nd(z):
        return (z**2 - 1) * _base_func(z)

    def _grad_3rd(z):
        return (3*z - z**2) * _base_func(z)

    def _grad_4th(z):
        return (3 - 6*z**2 + z**4) * _base_func(z)

    #print(1, _grad_1st(z))
    #print(2, _grad_2nd(z))
    #print(3, _grad_3rd(z))
    #print(4, _grad_4th(z))

    if order == 4:
        return _grad_4th(z) * _factor()
    else:
        raise NotImplementedError()

def _ass_func_tanh_manual(z, d):
    """The associated function :math:`\psi` for a Tanh activation.

    This function uses manually computed derivatives and is not generally
    applicable.

    Args:
        (....): See docstring of function :func:`_ass_func`.

    Returns:
        (torch.Tensor): Returns the output of :math:`\psi(z)`.
    """
    r = d % 2

    def _exp_func(z):
        return torch.exp(-z**2 / 2)

    def _sin_func(z):
        return torch.sin(z * np.pi / 2)

    def _cos_func(z):
        return torch.cos(z * np.pi / 2)

    def _factor():
        return torch.Tensor([2**(-(d+r)/2) * np.pi**(-(d-r+2) / 2) * \
                             np.exp(np.pi**2 / 8)])

    order = d - r + 2

    def _grad_1st(z):
        return _exp_func(z) * (-z * _sin_func(z) + (np.pi / 2) * _cos_func(z))

    def _grad_2nd(z):
        return _exp_func(z) * (_sin_func(z) * (z**2 - np.pi**2 / 4 - 1) - \
                               _cos_func(z) * (np.pi * z))

    #print(1, _grad_1st(z))
    #print(2, _grad_2nd(z))

    if order == 2:
        return _grad_2nd(z) * _factor()
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    pass


