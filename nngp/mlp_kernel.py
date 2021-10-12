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
# @title          :nngp/mlp_kernel.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/19/2021
# @version        :1.0
# @python_version :3.8.5
r"""
The Kernel of infinite-width MLPs
---------------------------------

The module :mod:`nngp.mlp_kernel` implements ways to approximate the kernel
of an infinite width MLP (given the number of hidden layers and the activation
function).

The implementations in this module are inspired by the paper:

    `Lee et al. <https://arxiv.org/abs/1711.00165>`__,
    Deep Neural Networks as Gaussian Processes, 2017.

As described in the paper (cf. Eq. 4), the Kernel of an infinite-width MLP can
be recursively computed as

.. math::
    :label: kernel-recursion

    k^l(x, x') = \sigma^2_b + \sigma^2_w \
    \mathbb{E}_{z_i^{l-1} \sim \mathcal{GP}\big(0, K^{l-1} \big)} \
    \big[ \phi\big( z_i^{l-1}(x) \big) \phi\big( z_i^{l-1}(x') \big) \big]

with nonlinearity :math:`\phi(\cdot)` and a prior distribution over weights/
biases with zero mean and variance
:math:`\sigma^2_w / N_l` and :math:`\sigma_b^2`, respectively. Note, that
:math:`z_i^{l-1} \sim \mathcal{GP}\big(0, K^{l-1} \big)` is a 2D vector with
entries :math:`z_i^{l-1}(x)` and :math:`z_i^{l-1}(x')`. The recursion starts at

.. math::
    :label: kernel-init

    k^0(x, x') = \sigma^2_b + \frac{\sigma^2_w}{d_{\text{in}}} \
        \langle x, x' \rangle

with :math:`d_{\text{in}}` denoting the input dimensionality. Please refer to
the paper for more details.

In section 2.5, `Lee et al. <https://arxiv.org/abs/1711.00165>`__ describe an
efficient implementation of the above recursion that builds on a coarse
approximation using bilinear interpolation for query points inside a predefined
grid. In addition to this efficient implementation, this module also provides
the computationally more expensive direct estimation of the above recursion
using Monte-Carlo.
"""
import math
import torch
from torch.distributions import Normal, MultivariateNormal
import traceback
from warnings import warn

class MLPKernel:
    r"""A kernel function estimator for an infinite-width MLP.

    Args:
        n_layer (int): Number of (infinite-width) hidden layers.
        sigma2_w (float): The variance of weights :math:`\sigma^2_w`.
        sigma2_b (float): The variance of biases :math:`\sigma^2_b`.
        nonlinearity (func): The used nonlinearity.
    """
    def __init__(self, n_layer=1, sigma2_w=1., sigma2_b=1.,
                 nonlinearity=torch.nn.ReLU()):
        self._n_layer = n_layer
        self._sigma2_w = sigma2_w
        self._sigma2_b = sigma2_b
        self._nonlinearity = nonlinearity

        # Attributes for building the lookup table, see `kernel_efficient`.
        self._lookup = None
        self._u_max = 100
        self._s_min = 1e-8
        self._s_max = 100
        self._n_u = 101
        self._n_s = 151
        self._n_c = 131

    def kernel_efficient(self, X):
        r"""Efficient estimation of the kernel.

        This method uses the algorithm described in section 2.5 of
        `Lee et al. <https://arxiv.org/abs/1711.00165>`__.

        The settings for building the lookup table can be set via method
        :meth:`lookup_settings`.

        Note:
            This method assumes that all inputs have been preprocessed to have
            identical norm, i.e.,
            :math:`\langle x, x \rangle = \langle x', x' \rangle`!

        Args:
            X (torch.Tensor or tuple): See docstring of function
                :func:`init_kernel`.

        Returns:
            (torch.Tensor): See docstring of function :func:`init_kernel`.
        """
        if self._lookup is None:
            self._build_lookup_table()

        return self._kernel_comp_wrapper(X, comp_type='efficient')

    def kernel_mc(self, X, num_samples=100, parallel_comp=True):
        """Kernel estimation via Monte-Carlo sampling.

        This method estimates the kernel values by recursively approximating
        Eq :eq:`kernel-recursion` using Monte-Carlo sampling.

        Args:
            (....): See docstring of method :meth:`kernel_efficient`.
            num_samples (int): The number of MC samples used for every expected
                value that needs to be estimated.
            parallel_comp (bool): Whether parallel computation should be
                utilized. This option should always be ``True`` and is just
                provided for debugging purposes.

        Returns:
            (torch.Tensor): See docstring of function :func:`init_kernel`.
        """
        return self._kernel_comp_wrapper(X, comp_type='mc',
            num_samples=num_samples, parallel_comp=parallel_comp)

    def kernel_analytic(self, X, nl_type=None):
        """Computing the kernel values analytically.

        Certain nonlinearities allow an analytic evaluation of Eq.
        :eq:`kernel-recursion`.

        Args:
            (....): See docstring of method :meth:`kernel_efficient`.
            nl_type (str, optional): If not specified, the function will try to
                automatically determine the type of non-linearity and raise an
                exception if not possible. However, the user can also explicitly
                determine which type of non-linearity is used. The following
                types are available:

                  - ``'relu'``: A ReLU nonlinearity. The kernel is computed
                    using Eq. 11 from
                    `Lee et al. <https://arxiv.org/abs/1711.00165>`__.
                  - ``'one_sided_poly_0'``: One-sided polynomial of degree 0
                    (i.e., step-function). See function
                    :func:`one_sided_polynomial_recursion`
                  - ``'one_sided_poly_1'``: Same as ``'one_sided_poly_0'`` for
                    degree 1 (i.e., ReLU).
                  - ``'one_sided_poly_2'``: Same as ``'one_sided_poly_0'`` for
                    degree 2
                  - ``'erf'``: An error function nonlinearity. The kernel is
                    computed using Eq. 10 from
                    `Pang et al. <https://arxiv.org/abs/1806.11187>`__.
                  - ``'cos'``: A cosine nonlinearity. The analytic kernel is
                    only available for a 1-hidden layer network and computed
                    using Eq. 36 from
                    `Pearce et al. <https://arxiv.org/abs/1905.06076>`__.

        Returns:
            (torch.Tensor): See docstring of function :func:`init_kernel`.
        """
        if nl_type is None:
            if isinstance(self._nonlinearity, torch.nn.ReLU):
                nl_type = 'relu'
            elif self._nonlinearity == torch.erf:
                nl_type = 'erf'
            elif self._nonlinearity == torch.cos:
                nl_type = 'cos'
            else:
                raise RuntimeError('Analytic kernel expression for ' + \
                    'nonlinearity of type %s could not be determined.' % \
                    (type(self._nonlinearity)))

        assert nl_type in ['relu', 'one_sided_poly_0', 'one_sided_poly_1',
                           'one_sided_poly_2', 'erf', 'cos']

        if nl_type == 'cos' and self._n_layer > 0:
            if self._n_layer != 1:
                raise NotImplementedError('The analytic kernel for the ' +
                    'cosine non-linearity is only available for 1-hidden ' +
                    'layer.')
            return cosine_1l_kernel(X, sigma2_w=self._sigma2_w,
                                    sigma2_b=self._sigma2_b)

        return self._kernel_comp_wrapper(X, comp_type='analytic',
                                         nl_type=nl_type)

    def lookup_settings(self, n_u=101, n_s=151, n_c=131, s_min=1e-8, s_max=100,
                        u_max=100):
        """Set the settings for the lookup table used by method
        :meth:`kernel_efficient`.

        If these settings differ from the current ones, the lookup table will
        be recomputed upon the next call to method :meth:`kernel_efficient`.

        Args:
            n_u (int): The number of linearly-spaced pre-activations between
                ``-u_max`` and ``u_max``.
            n_s (int): The number of linearly-spaces variances in
                ``[s_min, ..., s_max]``.
            n_c (int): The number of linearly-space correlations to consider
                between ``[-1, ..., 1]``.
            s_min (float): The minimum variance.
            s_max (float): The maximum variance.
            u_max (float, optional): The maximum pre-activation. Note, that the
                original algorithm expects ``s_max < u_max**2``.

                If not provided, it will be set to twice the standard deviation:
                    ``u_max = 2*sqrt(s_max)``.
        """
        if self._n_u != n_u or self._n_s != n_s or self._n_c != n_c or \
                self._s_min != s_min or self._s_max != s_max or \
                self._u_max != u_max:
            # Reset lookup table.
            self._lookup = None

            self._n_u = n_u
            self._n_s = n_s
            self._n_c = n_c
            self._s_min = s_min
            self._s_max = s_max
            self._u_max = u_max

    def _kernel_comp_wrapper(self, X, comp_type='mc', **kwargs):
        """Wrapper method for kernel computation.

        Args:
            (....): See docstring of method :meth:`kernel_efficient`.
            comp_type (str): ``'mc'``, ``'efficient'`` or ``'analytic'``

        Returns:
            See docstring of method :meth:`kernel_efficient`.
        """
        assert comp_type in ['mc', 'analytic', 'efficient']
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
            n = X1.shape[0]

            K = init_kernel((X1, X2), sigma2_w=self._sigma2_w,
                            sigma2_b=self._sigma2_b)
            if self._n_layer == 0:
                return K

            # We need to build a batch of 2x2 matrices comprising for the i-th
            # tuple `(x_i1, x_i2) `the kernel values `K(x_i1, x_i1)`,
            # `K(x_i2, x_i2)` and `K(x_i1, x_i2)`
            K_ij = K
            K_ii = init_kernel((X1, X1), sigma2_w=self._sigma2_w,
                               sigma2_b=self._sigma2_b)
            K_jj = init_kernel((X2, X2), sigma2_w=self._sigma2_w,
                               sigma2_b=self._sigma2_b)

            K_init = K_ij.view(-1, 1, 1).repeat(1, 2, 2)
            K_init[:, 0, 0] = K_ii
            K_init[:, 1, 1] = K_jj

            if comp_type == 'mc' and kwargs['parallel_comp']:
                K = self._kernel_mc_recursion_parallel(K_init, 1, self._n_layer,
                    kwargs['num_samples'])
                K = K[:, 0, 1]
                return K

            elif comp_type == 'analytic' and kwargs['nl_type'] == 'relu':
                # Compute the full 2x2 kernel matrix, but omit it
                # except for the off-diagonal element.
                K = relu_recursion_lee(K_init, 1, self._n_layer,
                    sigma2_w=self._sigma2_w, sigma2_b=self._sigma2_b)
                K = K[:, 0, 1]
                return K

            elif comp_type == 'analytic' and kwargs['nl_type'] == 'erf':
                K = erf_recursion(K_init, 1, self._n_layer,
                    sigma2_w=self._sigma2_w, sigma2_b=self._sigma2_b)
                K = K[:, 0, 1]
                return K

            else:
                # FIXME Batch-wise processing of kernel matrices not supported!
                # Compute the kernel value for each tuple separately.
                if n > 1000:
                    warn('No efficient implementation available!')

                K = torch.empty(n).to(X1.device)
                X_curr = torch.empty(2, X1.shape[1])
                for i in range(n):
                    X_curr[0,:] = X1[i,:]
                    X_curr[1,:] = X2[i,:]
                    K_0 = K_init[i, :, :]

                    if comp_type == 'mc':
                        assert not kwargs['parallel_comp']
                        # Compute the complete 2x2 kernel matrix `K^{L-1}`.
                        K_Lm1 = self._kernel_mc_recursion(K_0, 1,
                                self._n_layer-1, kwargs['num_samples'])
                        # Compute scalar `K_L`for the i-th tuple (proper scaling
                        # is done below).
                        K[i] = self._expected_value_mc(K_Lm1,
                                                       kwargs['num_samples'])
                    elif comp_type == 'analytic':
                        if kwargs['nl_type'].startswith('one_sided_poly_'):
                            K_tmp = one_sided_polynomial_recursion(X, None, 1,
                                self._n_layer,
                                poly_degree=int(kwargs['nl_type'][-1]),
                                sigma2_w=self._sigma2_w,
                                sigma2_b=self._sigma2_b)
                            K[i] = K_tmp[0, 1]
                        else: # If we end up here, we have a bug!
                            raise NotImplementedError()
                    elif comp_type == 'efficient':
                        K_tmp = self._kernel_efficient_recursion(K_0, 1,
                                                                 self._n_layer)
                        K[i] = K_tmp[0, 1]

                if comp_type == 'mc': # Note, we used `_expected_value_mc`.
                    K = self._sigma2_b + self._sigma2_w * K
                return K

        else:
            K_0 = init_kernel(X, sigma2_w=self._sigma2_w,
                              sigma2_b=self._sigma2_b)
            if comp_type == 'mc':
                if kwargs['parallel_comp']:
                    return self._kernel_mc_recursion_parallel(K_0, 1,
                        self._n_layer, kwargs['num_samples'])
                else:
                    return self._kernel_mc_recursion(K_0, 1, self._n_layer,
                                                     kwargs['num_samples'])
            elif comp_type == 'analytic':
                if kwargs['nl_type'] == 'relu':
                    return relu_recursion_lee(K_0, 1, self._n_layer,
                        sigma2_w=self._sigma2_w, sigma2_b=self._sigma2_b)
                elif kwargs['nl_type'] == 'erf':
                    return erf_recursion(K_0, 1, self._n_layer,
                        sigma2_w=self._sigma2_w, sigma2_b=self._sigma2_b)
                elif kwargs['nl_type'].startswith('one_sided_poly_'):
                    return one_sided_polynomial_recursion(X, None, 1,
                        self._n_layer, poly_degree=int(kwargs['nl_type'][-1]),
                        sigma2_w=self._sigma2_w, sigma2_b=self._sigma2_b)
            elif comp_type == 'efficient':
                    return self._kernel_efficient_recursion(K_0, 1,
                                                            self._n_layer)

    def _kernel_mc_recursion_parallel(self, K_prev, curr_layer, end_layer,
                                      num_samples):
        r"""Compute an MC estimate of the current kernel matrix.

        This method should do the same as method :meth:`_kernel_mc_recursion`,
        but the computation of kernel entries is parallelized.

        Diagonal entries of the kernel matrix will be computed under the
        assumption of singular covariance matrices (see
        :meth:`_expected_value_mc`) and all off-diagonal elements are computed
        using the multivariate reparametrization trick

        .. math::
            z \sim \mathcal{N}(\mu, \Sigma) \Leftrightarrow \
            z = \mu + L\epsilon \quad \text{ with } \
            \epsilon \sim \mathcal{N}(0, I), \Sigma = L L^T
        
        Therefore, we utilize the function :func:`torch.cholesky`, which allows
        computing the Cholesky decomposition for a batch of matrices. However,
        if the Cholesky decomposition for any off-diagonal element fails, this
        method will fall back to :meth:`_kernel_mc_recursion`.

        Args:
            (....): See docstring of method :meth:`_kernel_mc_recursion`.

        Returns:
            (torch.Tensor): The kernel matrix :math:`K^L`.
        """
        if curr_layer > end_layer:
            return K_prev

        bs = K_prev.shape[:-2] # Batch shape.

        K = torch.empty_like(K_prev)

        n = K.shape[-1]

        ### Compute diagonal (singular covariance matrix).
        diag_dist = Normal(0., torch.sqrt(torch.diagonal(K_prev, dim1=-2,
                                                         dim2=-1)))
        # k == 1, thus x1 == x2
        x = diag_dist.sample((num_samples,))
        torch.diagonal(K, dim1=-2, dim2=-1)[:] = torch.mean( \
            self._nonlinearity(x)**2, dim=0)

        ### Compute off-diagonal elements.
        # For each off-diagonal element K_ij, we have to construct a covariance
        # matrix C_ij = [[K_ii, K_ij], [K_ij, K_jj]].
        triu_inds = torch.triu_indices(n, n, offset=1).to(K_prev.device)
        assert triu_inds.shape[0] == 2
        # Each column in `tril_inds` contains indices ij. For those, we now
        # create tuples iijj and ijij.
        inds_ijij = triu_inds.repeat(2, 1)
        inds_iijj = torch.cat([triu_inds[0,:].repeat(2, 1),
                               triu_inds[1,:].repeat(2, 1)], dim=0)

        C_inds = torch.cat([inds_iijj.view(1, -1), inds_ijij.view(1, -1)],
                           dim=0)

        C = K_prev[..., C_inds[0, :], C_inds[1, :]]
        C = C.view(*bs, 2, 2, -1).permute(*list(range(len(bs))), -1, -3, -2)

        try:
            L = torch.linalg.cholesky(C)
            #L = torch.cholesky(C, upper=False)
        except:
            # Since the cholesky failed, either some eigenvalues are zero
            # (singular -> positive semi-definite) or even negative (not
            # positive definite). We will try to add noise perturbations to
            # these matrices in order to make them positive definite. See
            # https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
            #traceback.print_exc()

            min_evs = torch.linalg.eigvalsh(C).min(dim=-1)[0]
            corrections = -min_evs + 1e-5
            mask = min_evs > 0
            corrections[mask] = 0


            corrections = corrections.view(*corrections.shape, 1, 1). \
                repeat(*[1]*len(corrections.shape),2,2)
            # Exlude off-diagonal elements.
            corrections[..., [0, 1], [1, 0]] = False

            # Add corrections to diagonal.
            C += corrections

            try:
                L = torch.linalg.cholesky(C)
            except: # Shouldn't happen!
                #traceback.print_exc()
                warn('Singular matrix occured in off-diagonal kernel ' +
                     'element computation. Computing kernel matrix elements ' +
                     'individually.')
                return self._kernel_mc_recursion(K_prev, curr_layer, end_layer,
                    num_samples, continue_with_parallel=True)

        # Reparametrization trick.
        std_normal = Normal(0, torch.tensor(1.).to(K_prev.device))
        eps = std_normal.sample((2, num_samples))

        # GP samples.
        x = self._nonlinearity(torch.matmul(L, eps))
        K_off_diag = torch.mean(x.prod(dim=-2), dim=-1)

        K[..., triu_inds[0,:], triu_inds[1,:]] = K_off_diag

        # Copy the upper triangular part to the lower triangular part.
        linds = torch.tril_indices(n, n, offset=-1)
        K[..., linds[0,:], linds[1,:]] = \
            torch.transpose(K, -2, -1)[..., linds[0,:], linds[1,:]]

        # Finish the kernel computation (so far, we only computed the
        # expected values).
        K = self._sigma2_b + self._sigma2_w * K

        return self._kernel_mc_recursion_parallel(K, curr_layer+1, end_layer,
                                                  num_samples)

    def _kernel_mc_recursion(self, K_prev, curr_layer, end_layer, num_samples,
                             continue_with_parallel=False):
        r"""Compute an MC estimate of the current kernel matrix.

        This function will recursively compute the kernel :math:`K^l` given
        :math:`K^{l-1}` according to Eq. :eq:`kernel-recursion` until
        :math:`l = L`.

        Args:
            (....): See docstring of method :meth:`kernel_mc`.
            K_prev (torch.Tensor): The kernel matrix :math:`K^{l-1}`.

                This function supports processing batches of kernel matrices!
            curr_layer (int): The current layer index :math:`l`.
            end_layer (int): The stopping criterion :math:`L`.
            continue_with_parallel (bool): If ``True``, the recursion continues
                with method :meth:`_kernel_mc_recursion_parallel`.

        Returns:
            (torch.Tensor): The kernel matrix :math:`K^L`.
        """
        if curr_layer > end_layer:
            return K_prev

        bs = K_prev.shape[:-2] # Batch shape.
        n = K_prev.shape[-1]

        # Collapse all batch dimensions into 1 dimension (ew later undo this.)
        K_prev = K_prev.view(-1, n, n)
        K = torch.empty_like(K_prev)

        # In order to compute the full kernel matrix `K`, we have to iterate
        # over every index `K_ij` in the lower (or upper) triangular matrix
        # (incl. the diagonal) if `K_pref` and build a Bivariate Gaussian
        # distribution (requiring entries `ii`, `ij` and `jj` from `K_pref`).
        C = torch.empty(2,2).to(K_prev.device) # Bivariate covariance matrix.
        for b in range(K_prev.shape[0]):
            for i in range(n):
                for j in range(i, n):
                    C[(0,0,1,1), (0,1,0,1)] = K_prev[b, (i,i,j,j), (i,j,i,j)]

                    K[b,i,j] = self._expected_value_mc(C, num_samples,
                        is_singular=i==j, k=1.)

        K = K.view(*bs, n, n)

        # Copy the upper triangular part to the lower triangular part.
        linds = torch.tril_indices(n, n, offset=-1)
        K[..., linds[0,:], linds[1,:]] = \
            torch.transpose(K, -2, -1)[..., linds[0,:], linds[1,:]]

        # Finish the kernel computation (so far, we only computed the
        # expected values).
        K = self._sigma2_b + self._sigma2_w * K

        if continue_with_parallel:
            return self._kernel_mc_recursion_parallel(K, curr_layer+1,
                end_layer, num_samples)
        return self._kernel_mc_recursion(K, curr_layer+1, end_layer,
                                         num_samples)

    def _expected_value_mc(self, C, num_samples, is_singular=None, k=1.):
        r"""Compute the expected value in Eq. :eq:`kernel-recursion`.

        This method estimates

        .. math::

            \mathbb{E}_{z_i^{l-1} \sim \mathcal{GP}\big(0, K^{l-1} \big)} \
            \big[ \phi\big( z_i^{l-1}(x) \big) \phi\big( z_i^{l-1}(x') \big) \
            \big]

        Args:
            (....): See docstring of method :meth:`kernel_mc`.
            C (torch.Tensor): A 2x2 covariance matrix.
            is_singular (bool, optional): Whether the covariance matrix ``C``
                is singular.

                If the construction of a bivariate Gaussian fails, then ``C``
                will automatically be expected to be singular.
            k (float): The scalar that connects the first and second column of
                ``C``: :math:`\vec{c}_1 = k \vec{c}_2`.

        Returns:
            (torch.Tensor): The scalar MC-estimate of the expected value.
        """
        if not is_singular:
            try:
                mvg = MultivariateNormal(torch.zeros(2).to(C.device), C)
            except:
                ii = -1
                while True:
                    ii += 1
                    evs = torch.linalg.eigvalsh(C)

                    if torch.any(torch.isclose(evs, torch.zeros_like(evs))):
                        is_singular = True
                        k = C[0,0] / C[0, 1]
                        break

                    # We assume the matrix is not positive definite (i.e., some
                    # eigenvalues are negative).
                    # We try adding a small multiple of the identity matrix to
                    # overcome  numerical issues - see step 1 here:
                    # https://juanitorduz.github.io/multivariate_normal/
                    #C += 0.0001 * torch.eye(2).to(C.device)
                    # To avoid excessive looping, we correct by the smallest
                    # eigenvalue.
                    # https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
                    C += (-evs.min() + 1e-5) * torch.eye(2).to(C.device)

                    try:
                        mvg = MultivariateNormal(torch.zeros(2).to(C.device), C)
                        break
                    except:
                        continue

        # Here is a good description of how to handle the singular case.
        # https://stackoverflow.com/a/27641799
        # As described, in the singular case the matrix can be written as
        # | k*k*c   k*c |
        # |  k*c     c  |
        # Thus, we can sample x_2 ~ N(0, c) and deterministically compute
        # x_1 = k * x_2 (note, that both are zero-mean).
        # Alternatively, we could do
        #   eps ~ N(0, 1)
        #   x_1 = k * sqrt(c) * eps
        #   x_2 = sqrt(c) * eps
        if is_singular:
            mvg = Normal(0., torch.sqrt(C[1,1]))
            x2 = mvg.sample((num_samples,))
            x1 = k * x2

            return torch.mean(self._nonlinearity(x1) * self._nonlinearity(x2))

        else:
            x = self._nonlinearity(mvg.sample((num_samples,)))
            return torch.mean(x.prod(dim=1))

    def _build_lookup_table(self):
        """Build lookup table for efficient kernel computation."""
        if self._u_max is None:
            self._u_max = 2 * math.sqrt(self._s_max)
        elif self._s_max >= self._u_max**2:
            raise ValueError('"s_max < u_max^2" condition violated.')

        self._c = torch.linspace(-1, 1, self._n_c)
        self._s = torch.linspace(self._s_min, self._s_max, self._n_s)
        u = torch.linspace(-self._u_max, self._u_max, self._n_u)
        phi_u = self._nonlinearity(u)

        ### Handle all non-singular cases ###
        # I.e., -1 < covariance values < 1.

        ### We first build all possible 2D covariance matrices.
        s_rep = self._s.view(-1, 1).repeat(1, self._n_c-2)
        # Compute all possible products s_i * c_j, except if c_j is -1 or 1.
        sc_prods = s_rep * self._c[1:-1][None, :]
        # sc_prods has shape [n_s, n_c-2]
        # Build 2x2 covariance matrix for every i,j.
        P = s_rep.view(self._n_s, self._n_c-2, 1, 1).repeat(1, 1, 2, 2)
        P[:,:,0,1] = sc_prods
        P[:,:,1,0] = sc_prods
        # Transform covariance matrices into precision matrices.
        P = torch.linalg.inv(P)

        off_inds = torch.tril_indices(self._n_u, self._n_u, offset=-1)
        # All vectors [u_a, u_b] with a != b
        u_offdiag = u[off_inds]
        # For all [u_a, u_b], compute the (unnormalized) density for each
        # covariance matrix.
        p_ab_offdiag = torch.exp(-.5 * (u_offdiag * \
            torch.matmul(P, u_offdiag)).sum(dim=2))
        # p_ab_offdiag has shape [n_s, n_c-2, n_od] where n_od is the number of
        # off-diag elements.

        u_diag = u.repeat(2, 1)
        p_ab_diag = torch.exp(-.5 * (u_diag * \
                                     torch.matmul(P, u_diag)).sum(dim=2))
        # p_ab_diag has shape [n_s, n_c-2, n_u]

        norm_factors = 2 * p_ab_offdiag.sum(dim=2) + p_ab_diag.sum(dim=2)
        F = (phi_u**2 * p_ab_diag).sum(dim=2) + (phi_u[off_inds].prod(dim=0) * \
                                                 p_ab_offdiag).sum(dim=2) * 2
        F /= norm_factors

        ### Handle all singular cases ###
        # I.e., the cases if the correlation is 1 or -1.

        # If the correlation is 1, then the exp-factor is zero whenever
        # u_a != u_b.
        # If the correlation is -11, then the exp-factor is zero whenever
        # u_a != -u_b.
        # Otherwise, the exp factor is simply taken from a 1D Gaussian:
        # exp(- 1/2 u_a**2 / s_i)
        p_aa = torch.exp(-.5 * u[None,:]**2 / self._s[:,None])
        norm_factors = p_aa.sum(dim=1)

        # Note, that `phi_u * phi_u.flip(0) = - phi_u**2
        phi_u2 = phi_u**2

        F_cp1 = (phi_u2[None,:] * p_aa).sum(dim=1) / norm_factors
        F_cm1 = - F_cp1

        F = torch.cat([F_cm1.view(-1, 1), F, F_cp1.view(-1, 1)], dim=1)

        # F has shape [n_s, n_c].
        self._lookup = F

    def _kernel_efficient_recursion(self, K_prev, curr_layer, end_layer):
        """Compute the current kernel matrix via a lookup table.

        Note:
            ``K_prev`` is expected to have identical diagonal entries due to the
            required data preprocessing of method :meth:`kernel_efficient`.

        Args:
            (....): See docstring of method :meth:`_kernel_mc_recursion`.

        Returns:
            (torch.Tensor): The kernel matrix :math:`K^L`.
        """
        if len(K_prev.shape) != 2: # TODO
            raise ValueError('Function cannot process batches of kernel ' +
                             'matrices.')

        if curr_layer > end_layer:
            return K_prev

        K = torch.empty_like(K_prev)
        n = K.shape[-1]

        # According to the paper, the kernel entries K^l(x, x') are computed
        # via bilinear interpolation using the lookup table.
        # The s-value is given by K^{l-1}(x, x) and the c-value by
        # K^{l-1}(x, x') / K^{l-1}(x, x).
        linds = torch.tril_indices(n, n)
        # Note, in principle all diagonal entries are identical and thus all
        # s-values are identical.
        s_val = K_prev[linds[0,:], linds[0,:]] # K^{l-1}(x, x)
        c_val = K_prev[linds[0,:], linds[1,:]] / s_val

        ### NUMPY IMPLEMENTATION - START ###
        #import numpy as np
        #from scipy import interpolate
        #from scipy.interpolate import griddata
        #s_np = self._s.detach().cpu().numpy().squeeze()
        #c_np = self._c.detach().cpu().numpy().squeeze()

        # Option 1
        #f_np = interpolate.interp2d(s_np, c_np,
        #    self._lookup.T.detach().cpu().numpy(), kind='linear')
        #K[linds[0,:], linds[1,:]]  = torch.Tensor([float(f_np(a, b)) \
        #                                           for a, b in \
        #    zip(s_val.detach().cpu().numpy(), c_val.detach().cpu().numpy())])

        # Option 2
        #ss, cc = np.meshgrid(s_np, c_np)
        #K[linds[0,:], linds[1,:]] = torch.Tensor(griddata( \
        #    (ss.flatten(), cc.flatten()),
        #    self._lookup.T.detach().cpu().numpy().flatten(),
        #    (s_val.detach().cpu().numpy(), c_val.detach().cpu().numpy()),
        #    method='linear'))
        ### NUMPY IMPLEMENTATION - END ###

        ### PYTORCH BILINEAR INTERPOLATION - START ###
        # Find indices of nearest neighbors.
        sinds = torch.max(s_val[None, :] <= self._s[:, None], dim=0)[1]
        cinds = torch.max(c_val[None, :] <= self._c[:, None], dim=0)[1]
        # In case the smallest index larger than the searched value is 0.
        sinds[sinds==0] = 1
        cinds[cinds==0] = 1
        # First row becomes index of element smaller than considered value and
        # second row becomes index of element larger than considered value.
        sinds = sinds.repeat(2, 1)
        cinds = cinds.repeat(2, 1)
        sinds[0,:] -= 1
        cinds[0,:] -= 1

        svals = self._s[sinds]
        cvals = self._c[cinds]
        # We now have tuples (smin, smax) and (cmin, cmax).
        # We need to extract 4 lookup positions for these:
        # F(smin,cmin), F(smin,cmax), F(smax,cmin), F(smax,cmax).
        sinds = torch.cat([sinds[0,:].repeat(2, 1),
                           sinds[1,:].repeat(2, 1)], dim=0)
        cinds = cinds.repeat(2, 1)
        Fvals = self._lookup[(sinds, cinds)].T.view(-1, 2, 2)

        # Perform bilinear interpolation.
        factor = 1 / ((svals[1,:]-svals[0,:]) * (cvals[1,:]-cvals[0,:]))
        sdiff = torch.empty_like(svals)
        sdiff[0, :] = svals[1, :] - s_val[None, :]
        sdiff[1, :] = s_val[None, :] - svals[0, :]
        cdiff = torch.empty_like(cvals)
        cdiff[0, :] = cvals[1, :] - c_val[None, :]
        cdiff[1, :] = c_val[None, :] - cvals[0, :]

        K[linds[0,:], linds[1,:]] = factor * torch.bmm(sdiff.T.view(-1, 1, 2),
            torch.bmm(Fvals, cdiff.T.view(-1, 2, 1))).view(-1)
        ### PYTORCH BILINEAR INTERPOLATION - END ###

        uinds = torch.triu_indices(n, n, offset=1)
        K[(uinds[0,:]), (uinds[1,:])] = K.T[uinds[0,:], uinds[1,:]]

        # The lookup interpolation only approximates F_phi.
        K = self._sigma2_b + self._sigma2_w * K

        return self._kernel_efficient_recursion(K, curr_layer+1, end_layer)


def init_kernel(X, sigma2_w=1., sigma2_b=1.):
    r"""Compute the initial kernel :math:`k^0(x, x')`.

    This function implements Eq. :eq:`kernel-init`.

    Args:
        (....): See docstring of class :class:`MLPKernel`.
        X (torch.Tensor or tuple): A tensor whose first dimension is the batch
            size. If inputs are scalar, then it is sufficient to pass a 1D
            tensor. Otherwise, a 2D tensor is expected where the second
            dimension reflects :math:`d_{\text{in}}`.

            If a tensor is passed with batch size :math:`B`, then the symmetric
            :math:`B \times B` kernel matrix that comprises all possible tuples
            is returned.

            If a tuple of two tensors with identical shape is passed, then only
            the kernel between corresponding pairs is returned.
    
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
    d_in = X1.shape[1]

    if X2 is None:
        K = torch.mm(X1, X1.T) # Gramian matrix
    else:
        K = torch.sum(X1 * X2, dim=1)

    return sigma2_b + sigma2_w / d_in * K

def cosine_1l_kernel(X, sigma2_w=1., sigma2_b=1.):
    r"""Compute the kernel :math:`k^1(x, x')` for an MLP with cosine activation.

    This function computes the kernel for a 1-hidden layer MLP with cosine
    activation function as derived in
    `Pearce et al. <https://arxiv.org/abs/1905.06076>`__ (see Eq. 36). The
    kernel is computed as

    .. math::

        k^1(x, x') = \sigma_b^2 + \frac{\sigma_w^2}{2} \bigg( \
            \exp \Big( - \
                \frac{\sigma_w^2 \lVert x - x' \rVert^2}{2 d_{\text{in}}} \
            \Big) + \exp \Big( - \
                \frac{\sigma_w^2 \lVert x + x' \rVert^2}{2 d_{\text{in}}} - \
            2 \sigma_b^2 \Big) \bigg)

    Args:
        (....): See docstring of function :func:`init_kernel`.

    Returns:
        (torch.Tensor): The kernel matrix :math:`K^L` for :math:`L=1`.
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
    d_in = X1.shape[1]

    if X2 is None:
        X_diff = X1.unsqueeze(1) - X1.unsqueeze(0)
        X_sum = X1.unsqueeze(1) + X1.unsqueeze(0)

        ed_diff = torch.sum(X_diff**2, dim=2)
        ed_sum = torch.sum(X_sum**2, dim=2)
    else:
        X_diff = X1 - X2
        X_sum = X1 + X2

        ed_diff = torch.sum(X_diff**2, dim=1)
        ed_sum = torch.sum(X_sum**2, dim=1)

    sigma2_w_in = sigma2_w / d_in
    K = torch.exp(- 0.5 * sigma2_w_in * ed_diff) + \
        torch.exp(- 0.5 * sigma2_w_in * ed_sum - 2 * sigma2_b)

    return sigma2_b + .5 * sigma2_w * K

def relu_recursion_lee(K_prev, curr_layer, end_layer, sigma2_w=1., sigma2_b=1.):
    r"""Analytic recursive kernel computation for ReLU MLPs.

    This implementation is based on Eq. 11 in
    `Lee et al. <https://arxiv.org/abs/1711.00165>`__.

    This equation is derived using Eq. :eq:`kernel-recursion` and noting that
    the term

    .. math::

        \mathbb{E}_{z_i^{l-1} \sim \mathcal{GP}\big(0, K^{l-1} \big)} \
        \big[ \phi\big( z_i^{l-1}(x) \big) \phi\big( z_i^{l-1}(x') \big) \big]

    can be considered as a dot-product between feature vectors
    :math:`\lim_{N \rightarrow \infty} \frac{1}{N} \langle f(x), f(x') \rangle`
    as studied in
    `Cho and Saul <https://papers.nips.cc/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf>`__.

    However, `Cho and Saul` assume that
    :math:`z_i^{l-1}(x) = \langle w^l_i, x_i^{l-1}(x) \rangle`
    with :math:`w^l_i \sim \mathcal{N}(0, I)`, while `Lee et al.` consider
    :math:`z_i^{l-1}(x) = \langle w^l_i, x_i^{l-1}(x) \rangle + b^l_i` with
    :math:`w^l_i \sim \mathcal{N}(0, \sigma_w^2/N_l I)` and
    :math:`b^l_i \sim \mathcal{N}(0, \sigma_b^2 I)`. The formula is derived
    using the simple equality:

    .. math::

        \langle \begin{bmatrix} \sigma_w / \sqrt{N_l} \
        \vec{x}_i^{l-1}(x)  \\ \sigma_b \end{bmatrix}, \
        \begin{bmatrix} \sigma_w / \sqrt{N_l} \
        \vec{x}_i^{l-1}(x')  \\ \sigma_b \end{bmatrix} \rangle = \
        \sigma_b^2 + \frac{\sigma_w^2}{N_l} \langle \vec{x}_i^{l-1}(x), \
        \vec{x}_i^{l-1}(x') \rangle

    For more details refer to function :func:`one_sided_polynomial_recursion`.

    Args:
        (....): See docstring of function :func:`init_kernel`.
        K_prev (torch.Tensor): The kernel matrix :math:`K^{l-1}`. The kernel
            matrix :math:`K^0` should be computed via :func:`init_kernel`.

            This function supports processing batches of kernel matrices!
        curr_layer (int): The current layer index :math:`l`.
        end_layer (int): The stopping criterion :math:`L`.

    Returns:
        (torch.Tensor): The kernel matrix :math:`K^L`.
    """
    if curr_layer > end_layer:
        return K_prev

    bs = K_prev.shape[:-2] # Batch shape.

    K = torch.empty_like(K_prev)

    # Compute `sqrt(K(x,x) * K(x',x'))
    sqrt_factor = torch.sqrt(torch.matmul( \
        torch.diagonal(K_prev, dim1=-2, dim2=-1).view(*bs, -1, 1), \
        torch.diagonal(K_prev, dim1=-2, dim2=-1).view(*bs, 1, -1)))

    theta_prev = torch.acos(K_prev / sqrt_factor)

    K = torch.sin(theta_prev) + (math.pi - theta_prev) * torch.cos(theta_prev)
    K = sigma2_b + (sigma2_w / (2 * math.pi)) * sqrt_factor * K

    return relu_recursion_lee(K, curr_layer+1, end_layer,
                              sigma2_w=sigma2_w, sigma2_b=sigma2_b)

def one_sided_polynomial_recursion(X, K_prev, curr_layer, end_layer,
                                   poly_degree=1, sigma2_w=1., sigma2_b=1.):
    r"""Analytic recursive kernel computation for MLPs with one-sided
    polynomial activation functions.

    This function uses the analytic kernels derived in `Cho and Saul
    <https://papers.nips.cc/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf>`__
    to compute the kernel function of an MLP (as in `Lee et al.`) with
    one-sided polynomial activation function. The main differences between the
    networks considered in `Cho and Saul` and `Lee et al.` are the following:

      - We consider MLPs with linear readout layer
      - We consider layers with bias vectors
      - Weight values may have a variance different from 1

    As a side note, `Cho and Saul` explicitly assume the weight prior to be
    Gaussian, while `Lee et al.` only specify the prior variance.

    To see how we combine the derivations of the two papers, let's start by
    recalling Eq. :eq:`kernel-recursion` for instance for the last layer
    :math:`L` (the linear output layer):

    .. math::

        k^L(x, x') = \mathbb{E} \big[ z_i^L(x) z_i^L(x') \big] = \
        \sigma^2_b + \sigma^2_w \
        \mathbb{E}_{z_i^{L-1} \sim \mathcal{GP}\big(0, K^{L-1} \big)} \
        \big[ \phi\big( z_i^{L-1}(x) \big) \phi\big( z_i^{L-1}(x') \big) \big]

    Consider a vector :math:`\vec{\epsilon} \sim \mathcal{N}(0, I)` that we
    decompose into

    .. math::

        \vec{\epsilon} = \begin{bmatrix}
           \vec{\epsilon}_w \\
           \epsilon_b
        \end{bmatrix}

    such that when using the reparametrization trick and assuming
    :math:`w^l_i \sim \mathcal{N}(0, \sigma_w^2/N_l I)` and
    :math:`b^l_i \sim \mathcal{N}(0, \sigma_b^2 I)`, we can write

    .. math::

        z_i^{L-1}(x) &= \langle \vec{w}^l_i, \vec{x}_i^{l-1}(x) \rangle + \
            b^l_i \\ \
            &= \langle \sigma_w/\sqrt{N_l} \vec{\epsilon}^l_{wi}, \
            \vec{x}_i^{l-1}(x) \rangle + \sigma_b \epsilon^l_{bi} \\ \
            &= \langle \vec{\epsilon}^l_{wi}, \sigma_w/\sqrt{N_l} \
            \vec{x}_i^{l-1}(x) \rangle + \sigma_b \epsilon^l_{bi} \\ \
            &= \langle \vec{\epsilon}^l_i, \begin{bmatrix} \sigma_w/\sqrt{N_l} \
            \vec{x}_i^{l-1}(x)  \\ \sigma_b \end{bmatrix} \rangle \\ \
            &\equiv \langle \vec{\epsilon}^l_i, \tilde{x}_i^{l-1}(x) \rangle

    with

    .. math::

        \tilde{x}_i^{l-1}(x) = \begin{bmatrix} \sigma_w/\sqrt{N_l} \
            \vec{x}_i^{l-1}(x)  \\ \sigma_b \end{bmatrix}

    The paper by `Cho and Saul` considers networks that compute their outputs as
    :math:`\phi\big(\langle \vec{\epsilon}^l_i,\tilde{x}_i^{l-1}(x)\big)\rangle`
    with :math:`\vec{\epsilon}^l_i \sim \mathcal{N}(0, I)` where
    :math:`\phi(\cdot)` is a one-sided polynomial (such as a ReLU).

    Specifically, they show that for the :math:`n`-th order arc-cosine kernel
    :math:`k_n(x, x')` it holds that

    .. math::
        :label: kernel-network-features

        k_n(\tilde{x}_i^{0}(x), \tilde{x}_i^{0}(x')) = \
        \lim_{N \rightarrow \infty} \frac{2}{N} \sum_{i=1}^N \
            \phi \big(\langle \vec{\epsilon}^1_i, \tilde{x}_i^{0}(x) \
                \rangle\big) \
            \phi \big(\langle \vec{\epsilon}^1_i, \tilde{x}_i^{0}(x') \
                \rangle \big)

    In addition, `Cho and Saul` consider stacking multiple such layers and
    derive an analytic expression for :math:`k^l_n(x, x')`.

    These analytic expressions only depend on dot-products of feature vectors.
    Now that we know that we can cast the networks of `Lee et al.` to the
    networks of `Cho and Saul` simply augmenting the feature vectors to obtain
    :math:`\tilde{x}_i^{l-1}(x)`, we just need to modify the dot-products
    (or kernels) using those augmented feature vectors.

    Let's first consider dot-products between input vectors
    :math:`\langle x, x' \rangle`

    .. math::

        \langle \tilde{x}_i^{0}(x), \tilde{x}_i^{0}(x') \rangle = \langle \
        \begin{bmatrix} \sigma_w/\sqrt{d_{in}} x \\ \sigma_b \end{bmatrix}, \
        \begin{bmatrix} \sigma_w/\sqrt{d_{in}} x' \\ \sigma_b \end{bmatrix} \
        \rangle = \sigma_b^2 + \frac{\sigma_w^2}{d_{in}} \langle x, x' \rangle

    Next, we consider how dot-products between (infinite-width) hidden
    representations change (As a side remark, `Lee et al.` denote linear
    activations by :math:`z` and non-linear ones by :math:`x`. We follow the
    same notation, considering augmented vectors :math:`\tilde{x}` still as
    non-linear activations as the actual transformation only happens when
    multiplying :math:`\tilde{x}` by
    :math:`\vec{\epsilon} \sim \mathcal{N}(0, I)`.)

    .. math::

        \langle \begin{bmatrix} \vdots \\ \tilde{x}_i^{l}(x) \\ \vdots \
        \end{bmatrix}, \
        \begin{bmatrix} \vdots \\ \tilde{x}_i^{l}(x') \\ \vdots \
        \end{bmatrix} \rangle &= \
        \langle \begin{bmatrix} \vdots \\ \sigma_w/\sqrt{N_l} x_i^{l}(x) \\ \
        \vdots \\ \sigma_b \end{bmatrix}, \
        \begin{bmatrix} \vdots \\ \sigma_w/\sqrt{N_l} x_i^{l}(x') \\ \
        \vdots \\ \sigma_b \end{bmatrix} \rangle \\ &= \
        \sigma_b^2 + \frac{\sigma_w^2}{N_l} \
        \langle \begin{bmatrix} \vdots \\ x_i^{l}(x) \\ \vdots \
        \end{bmatrix}, \
        \begin{bmatrix} \vdots \\ x_i^{l}(x') \\ \vdots \
        \end{bmatrix} \rangle \\ &= \
        \sigma_b^2 + \frac{\sigma_w^2}{N_l} \frac{N_l}{2} \
        k^{l-1}_n(x, x')

    where in the last line we used Eq. :eq:`kernel-network-features`.

    Note:
        If ``curr_layer=1``, then ``K_prev`` is ignored (can be passed as
        ``None``).

    Note:
        This method expects ``X`` to be a single tensor.

    Note:
        This function is identical to :func:`relu_recursion_lee` for
        ``poly_degree=1``.

    Args:
        (....): See docstring of function :func:`relu_recursion_lee` and
            :func:`init_kernel`.
        poly_degree (int): Polynomial degree :math:`n`.

    Returns:
        (torch.Tensor): The kernel matrix :math:`K^L`.
    """
    # FIXME The implementation of this function is correct, but the function
    # `relu_recursion_lee` is more efficient and seems more numerically stable
    # (and can easily extended to other polynomial degrees). Therefore, we
    # should deprecate this function at some point.
    if len(K_prev.shape) != 2:
        # TODO
        raise ValueError('Function cannot process batches of kernel matrices.')

    if end_layer == 0:
        # Special case. Network without non-linearities.
        return init_kernel(X, sigma2_w=sigma2_w, sigma2_b=sigma2_b)

    if curr_layer > end_layer:
        return K_prev

    if curr_layer == 0:
        # Skip that case, the base case is below.
        return one_sided_polynomial_recursion(X, K_prev, 1, end_layer,
            poly_degree=poly_degree, sigma2_w=sigma2_w, sigma2_b=sigma2_b)

    # Compute the angular dependence for the given degree n.
    def J_0(theta):
        return math.pi - theta
    def J_1(theta): # ReLU
        return torch.sin(theta) + (math.pi - theta) * torch.cos(theta)
    def J_2(theta):
        # I might be too stupid, but I didn't derive at the same analytic
        # expression for this term. Someone should double check that the
        # term is correct.
        raise NotImplementedError('Implementation should be checked!')
        return 3 * torch.sin(theta) * torch.cos(theta) + \
            (math.pi - theta) * (1 + 2 * torch.cos(theta)**2)

    if curr_layer == 1: # Base case.
        assert X.ndim in [1, 2]
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        d_in = X.shape[1]

        # Modify `X`: Original paper assumes layer computation f(x) = g(Vx)
        # with V drawn from a standard Gaussian. We assume layer computation
        # f(x) = Wx + b (see docstring).
        X *= math.sqrt(sigma2_w/d_in)
        X = torch.cat((X, torch.ones(X.shape[0], 1) * math.sqrt(sigma2_b)), 1)

        X_norm = torch.linalg.norm(X, dim=1).reshape(-1, 1)
        X_norm_gram = torch.mm(X_norm, X_norm.T)
        theta = torch.acos(torch.mm(X, X.T) / X_norm_gram)

        K = 1 / math.pi * X_norm_gram**poly_degree
        if poly_degree == 0:
            K *= J_0(theta)
        elif poly_degree == 1:
            K *= J_1(theta)
        elif poly_degree == 2:
            K *= J_2(theta)
        else:
            raise NotImplementedError()

    else:
        K_prev_diag = torch.diagonal(K_prev).view(-1, 1)
        K_prev_gram = torch.sqrt(torch.mm(K_prev_diag, K_prev_diag.T))
        theta = torch.acos(K_prev / K_prev_gram)

        K = 1 / math.pi * K_prev_gram**poly_degree

        if poly_degree == 0:
            K *= J_0(theta)
        elif poly_degree == 1:
            K *= J_1(theta)
        elif poly_degree == 2:
            K *= J_2(theta)
        else:
            raise NotImplementedError()

    K = sigma2_b + sigma2_w / 2 * K

    # We don't actually need to pass `X` anymore to higher kernel terms.
    return one_sided_polynomial_recursion(X, K, curr_layer+1, end_layer,
        poly_degree=poly_degree, sigma2_w=sigma2_w, sigma2_b=sigma2_b)

def erf_recursion(K_prev, curr_layer, end_layer, sigma2_w=1., sigma2_b=1.):
    r"""Analytic recursive kernel computation for MLPs using error functions as
    non-linearity.

    This implementation is based on Eq. 11 in
    `Williams <https://papers.nips.cc/paper/1996/file/ae5e3ce40e0404a45ecacaaf05e5f735-Paper.pdf>`__,
    which can be easily extended to the multilayer case, e.g., see Eq. 10 in
    `Pang et al. <https://arxiv.org/abs/1806.11187>`__.

    Williams draws input layer weights
    :math:`(b_i^0, W^0_{i0}, \dots, W^0_{id_\text{in}})^T` from a Gaussian
    :math:`\mathcal{N}(0, \Sigma)`, such that we can write
    :math:`K^0(x, x') = \tilde{x}^T \Sigma \tilde{x}'` with
    :math:`\tilde{x} = (1, x_1, \dots, x_{d_\text{in}})`.

    When using function :func:`init_kernel` to compute :math:`k^0` it is assumed
    that

    .. math::

        \Sigma = \begin{bmatrix} \sigma_b^2 & 0 & \hdots & 0 \\ \
            0 & \sigma_w^2 / d_\text{in} & \hdots & 0 \\ \
            0 & 0 & \ddots & 0 \\ \
            0 & 0 & \hdots & \sigma_w^2 / d_\text{in} \
        \end{bmatrix}

    and therefore :math:`k^0(x, x')` is identical to Eq. :eq:`kernel-init`.

    Combining these insights together with Eq. 9 and Eq. 11 from Williams'
    paper, we arrive at the recursive formula:

    .. math::

        k^l(x, x') = \sigma_b^2 + \frac{2 \sigma_w^2}{\pi} \sin^{-1} \
            \bigg( \
            \frac{2 k^{l-1}(x, x')}{\sqrt{\big( 1 + 2 k^{l-1}(x, x) \big) \
            \big( 1 + 2 k^{l-1}(x', x') \big)}}
            \bigg)

    Args:
        (....): See docstring of function :func:`relu_recursion_lee`.

    Returns:
        (torch.Tensor): The kernel matrix :math:`K^L`.
    """
    if curr_layer > end_layer:
        return K_prev

    bs = K_prev.shape[:-2] # Batch shape.

    K = torch.empty_like(K_prev)

    # Compute `sqrt((1 + 2*K(x,x)) * (1 + 2*K(x',x')))
    sqrt_factor = torch.sqrt(torch.matmul( \
        1 + 2*torch.diagonal(K_prev, dim1=-2, dim2=-1).view(*bs, -1, 1), \
        1 + 2*torch.diagonal(K_prev, dim1=-2, dim2=-1).view(*bs, 1, -1)))

    asin_ratio = torch.asin(2 * K_prev / sqrt_factor)

    K = sigma2_b + (2*sigma2_w / math.pi) * asin_ratio

    return erf_recursion(K, curr_layer+1, end_layer,
                         sigma2_w=sigma2_w, sigma2_b=sigma2_b)

if __name__ == '__main__':
    pass


