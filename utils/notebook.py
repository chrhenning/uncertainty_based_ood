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
# @title          :utils/notebook.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/27/2021
# @version        :1.0
# @python_version :3.8.10
"""
Helpers for Jupyter Notebooks
-----------------------------

A collection of helper functions for working with jupyter notebooks.
"""
from IPython.display import display, Markdown
import os
import pickle
from time import time
import torch
import tqdm
import traceback

from nngp.mlp_kernel import MLPKernel
from nngp.nngp import inference_with_isotropic_gaussian_ll, \
    gen_inference_kernels, cholesky_adaptive_noise
from nngp.rbf_net import RBFNetKernel
from nngp import standard_kernels
from utils.plotting import heatmap
from utils.misc import calc_regression_acc

def determine_rbf_net_kernel_func(config_dict):
    """Create a function handle for RBF net kernel computation.

    This function uses the given config to determine a function handle that
    translates inputs to kernel values by using the methods implemented in class
    :class:`nngp.rbf_net.RBFNetKernel`.

    Args:
        config_dict (dict): A dictionary with keys

            - ``'name'`` (str): Unique identifier.
            - ``'type'`` (str): ``'analytic'`` or ``'mc'``; determines kernel
              computation method.
            - ``'params'`` (dict): Constructor arguments of class
              :class:`nngp.rbf_net.RBFNetKernel`
            - ``'kernel_params'`` (dict): Keyword arguments for kernel
              computation method.

            Example:
                .. code-block:: python
    
                    {'name': 'rbf_net', 'type': 'analytic',
                     'params': {'n_lin_hidden_units': [], 'bandwidth': 1,
                                'sigma2_u': 100., 'sigma2_w': 200.,
                                'sigma2_b': 1.},
                     'kernel_params': {}}

    Returns:
        (func)
    """
    k = config_dict

    rbfnet_kernel = RBFNetKernel(**k['params'])
    if k['type'] == 'analytic':
        func = lambda X : rbfnet_kernel.kernel_analytic(X,
            **dict(k['kernel_params']))
    else:
        assert k['type'] == 'mc'
        func = lambda X : rbfnet_kernel.kernel_mc(X, **dict(k['kernel_params']))

    return func

def determine_mlp_kernel_func(config_dict):
    """Create a function handle for MLP kernel computation.

    This function uses the given config to determine a function handle that
    translates inputs to kernel values by using the methods implemented in class
    :class:`nngp.mlp_kernel.MLPKernel`.

    Args:
        config_dict (dict): See docstring of function
            :func:`determine_rbf_net_kernel_func`.

    Returns:
        (func)
    """
    k = config_dict

    mlp_kernel = MLPKernel(**k['params'])
    if k['type'] == 'analytic':
        func = lambda X : mlp_kernel.kernel_analytic(X,
            **dict(k['kernel_params']))
    elif k['type'] == 'efficient':
        func = lambda X : mlp_kernel.kernel_efficient(X,
            **dict(k['kernel_params']))
    else:
        assert k['type'] == 'mc'
        func = lambda X : mlp_kernel.kernel_mc(X, **dict(k['kernel_params']))

    return func

def determine_classic_kernel_func(config_dict):
    """Create a function handle for kernel computation of standard kernels.

    This function uses the given config to determine a function handle that
    translates inputs to kernel values by using the methods implemented in
    module :mod:`nngp.standard_kernels`.

    Note:
        The kernel function is determined via the identifier ``'kernel'`` in the
        dictionary ``params``.

    Args:
        config_dict (dict): See docstring of function
            :func:`determine_rbf_net_kernel_func`.

    Returns:
        (func)
    """
    k = config_dict

    assert 'kernel' in k['params'].keys()

    if k['params']['kernel'] == 'rbf':
        func = lambda X : standard_kernels.rbf(X, **dict(k['kernel_params']))
    else:
        raise RuntimeError()

    return func

def compute_kernel_values(kernel_configs, X_test, X_train, Y_train,
                          determine_kernel_func, test_bs=50, out_dir=None,
                          try_inference=True, heatmap_kwargs=None):
    """Compute kernel matrices required for inference.

    This function allows to easily compute the kernel  matrices / vectors
    :math:`K(X, X)`, :math:`K(X, x^*)` and :math:`K(x^*, x^*)` required for
    inference. Therefore, this function makes use of function
    :func:`nngp.nngp.gen_inference_kernels`.

    The computed kernel matrices will be added to the individual kernel
    dictionaries inside ``kernel_configs`` using the keys ``'K_train'``,
    ``'K_test'`` and ``'K_all'``.

    Args:
        kernel_configs (list): List of dictionaries, each configuring a kernel.
            See argument ``config_dict`` of function
            :func:`determine_rbf_net_kernel_func` for details.
        X_test (torch.Tensor): Test inputs :math:`x^*`.
        X_train (torch.Tensor): Train inputs :math:`X`.
        Y_train (torch.Tensor): Train targets.
        determine_kernel_func (func): A function handle, e.g., function
            :func:`determine_rbf_net_kernel_func` or
            :func:`determine_mlp_kernel_func`.
        test_bs (int): Batch size for chunking :math:`K(X, x^*)` and
            :math:`K(x^*, x^*)`, as these are huge matrices that might not be
            computable in one go. Currently, this option only effects Monte-
            Carlo computation of kernel values.
        out_dir (str, optional): If specified, the computed kernel values will
            be saved in this directory.
        try_inference (bool): If ``True``, inference via function
            :func:`nngp.nngp.inference_with_isotropic_gaussian_ll` will be
            tested, and results will be plotted.
        heatmap_kwargs (dict): If ``try_inference=True``, this specified the
            keyword arguments to be passed to the
            :func:`utils.plotting.heatmap` function.
    """
    for i, k_props in enumerate(kernel_configs):
        display(Markdown('Investiagting **%s**' % (k_props['name'])))

        start = time()
        # There might be too many test samples in order to compute them all in
        # parallel (note, when doing an MC estimate, then a matrix of size
        # [num_test, num_mc] has to be hold in memory).
        K_test = []
        K_all = []
        n_test = X_test.shape[0]
        n_processed = 0

        chunk_size = test_bs if k_props['type'] == 'mc' else n_test

        pbar = tqdm.notebook.tqdm(total=n_test)
        while n_processed < n_test:
            X_test_curr = X_test[n_processed:n_processed+chunk_size]
            K_train_curr, K_test_curr, K_all_curr = \
                gen_inference_kernels(X_train, X_test_curr,
                                      determine_kernel_func(k_props),
                                      compute_K_train=n_processed==0,
                                      full_K_test=False)
            if n_processed == 0:
                K_train = K_train_curr
            K_test.append(K_test_curr)
            K_all.append(K_all_curr)
            n_processed += X_test_curr.shape[0]

            pbar.update(n_processed)
            pbar.refresh()
        pbar.close()

        K_test = torch.cat(K_test, dim=0)
        K_all = torch.cat(K_all, dim=0)
        display(Markdown('Kernel computation took %f seconds.' \
                         % (time()-start)))

        k_props['K_train'] = K_train
        k_props['K_test'] = K_test
        k_props['K_all'] = K_all

        # Backup kernel matrices.
        if out_dir is not None:
            k_props['backup_path'] = os.path.join(out_dir,
                                                  '%s.pickle' % k_props['name'])
            with open(k_props['backup_path'], 'wb') as f:
                pickle.dump(k_props, f)

        if not try_inference:
            continue
        assert heatmap_kwargs is not None

        try:
            L, ll_var = cholesky_adaptive_noise(K_train)
            k_props['L_train'] = L
            k_props['ll_Var'] = ll_var

            display(Markdown('Used likelihood variance: %f' % (ll_var)))

            train_mean, train_var = \
                inference_with_isotropic_gaussian_ll(Y_train, K_train,
                    torch.diagonal(K_train), K_train, L_mat=L, var=ll_var)
            display(Markdown('Accuracy on training points: %.2f%%' % \
                              calc_regression_acc(train_mean, Y_train)))

            start = time()
            grid_mean, grid_var = \
                inference_with_isotropic_gaussian_ll(Y_train, K_train,
                    K_test, K_all, L_mat=L, var=ll_var)
            display(Markdown('Inference took %f seconds.' % (time()-start)))

            heatmap(grid_mean, title='Predictive Posterior Mean - %s' \
                    % (k_props['name']), **heatmap_kwargs)
            heatmap(torch.sqrt(grid_var), title='Predictive Posterior Std - %s' \
                    % (k_props['name']), **heatmap_kwargs)
            heatmap(torch.sqrt(K_test), title='Prior Predictive Std - %s' \
                    % (k_props['name']), **heatmap_kwargs)
        except:
            traceback.print_exc()
            display(Markdown('Inference with **%s** failed!' \
                             % (k_props['name'])))

def load_kernel_values(kernel_configs, out_dir=None, device=None):
    """Load kernel matrices.

    Loads the kernel matrices computed in :func:`compute_kernel_values` and
    checks for NaN entries. If NaN values are found, a warning is displayed and
    the entries are set to zero.

    The loaded kernel matrices will be added to the individual kernel
    dictionaries inside ``kernel_configs`` using the keys ``'K_train'``,
    ``'K_test'`` and ``'K_all'``.

    Args:
        (....): See docstring of function :func:`compute_kernel_values`. Note,
            ``out_dir`` might be ``None``, in which case it is assumed that the
            kernels are already inside the dictionary.
        device: PyTorch device.
    """
    for i, k_props in enumerate(kernel_configs):
        display(Markdown('Loading kernel for **%s** ...' % (k_props['name'])))

        # Load backup.
        if out_dir is None:
            if not ('K_train' in k_props.keys() and 'K_test' in k_props.keys() \
                    and 'K_all' in k_props.keys()):
                display(Markdown('**WARN** No kernel found for **%s**.' \
                                 % (k_props['name'])))
                continue
            k_props_backup = k_props
        else:
            kpath = os.path.join(out_dir, '%s.pickle' % k_props['name'])
            if not os.path.exists(kpath):
                display(Markdown('**WARN** No kernel found for **%s**.' \
                                 % (k_props['name'])))
                continue
            with open(os.path.join(out_dir,
                                   '%s.pickle' % k_props['name']), 'rb') as f:
                k_props_backup = pickle.load(f)

        if device is None:
            device = k_props_backup['K_train'].device

        k_props['K_train'] = k_props_backup['K_train'].to(device)
        k_props['K_test'] = k_props_backup['K_test'].to(device)
        k_props['K_all'] = k_props_backup['K_all'].to(device)

        def check_nan(key):
            if torch.any(torch.isnan(k_props[key])):
                display(Markdown('**%d NaN values in %s**' \
                    % (int(torch.isnan(k_props[key]).sum().item()), key)))
                display(Markdown('**WARN** Setting NaN values to zero!'))
                k_props[key][~torch.isnan(k_props[key])] = 0

        check_nan('K_train')
        check_nan('K_test')
        check_nan('K_all')

if __name__ == '__main__':
    pass
