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
# @title          :utils/plotting.py
# @author         :ch, fd
# @contact        :henningc@ethz.ch
# @created        :07/09/2021
# @version        :1.0
# @python_version :3.8.10
"""
Plotting Utilities
------------------
"""
from hypnettorch.data.special.regression1d_data import ToyRegression
from hypnettorch.utils import misc as hmisc
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from warnings import warn

def heatmap(value_grid, grid_X=None, grid_Y=None, title=None, show_title=True,
            vmin=None, vmax=None, num_levels=50, log_levels=False, crop=False,
            force_extend=None, xticks=None, yticks=None, cticks=None,
            ctick_lbls=None, data_in=None, data_trgt=None, samples_in=None,
            filename=None, out_dir=None, ts=30, lw=15, figsize=(12, 7)):
    """Plot a 2D-heatmap.

    Plotting function to plot, for instance, uncertainty across the input space.

    Args:
        value_grid (numpy.ndarray or torch.Tensor): A 2D map of temperature
            values (e.g., uncertainties).
        grid_X (numpy.ndarray): See return value ``grid_X`` of function
            :func:`utils.misc.eval_grid_2d`.
        grid_Y (numpy.ndarray): See return value ``grid_Y`` of function
            :func:`utils.misc.eval_grid_2d`.
        title (str, optional): Plot title.
        show_title (bool): Whether title should be shown.
        vmin (float, optional): Minimum temperature value.
        vmax (float, optional): Maximum temperature value.
        num_levels (int): Number of color levels.
        log_levels (bool): If ``True``, colorbar will be in log scale.
        crop (bool): Crop values in ``value_grid`` according to
            ``vmin``/``vmax``. If ``False``, the plotting range will still be
            according to ``vmin``/``vmax``, but the color bar will denote
            outside colors via an arrow (except ``force_extend`` is set).
        force_extend (str, optional): You may set the values ``'neither'``,
            ``'min'``, ``'max'``, ``'both'`` to determine the colorbar layout.
            If not set, the layout will be set automatically.
        xticks (list, optional)
        yticks (list, optional)
        cticks (list, optional): Colorbar ticks.
        ctick_lbls (list, optional): Labels corresponding to colorbar ticks.
        data_in (numpy.ndarray, optional): 2D array of training inputs of shape
            ``[B, 2]``, where ``B`` denotes the batch size.
        data_trgt (numpy.ndarray, optional): 2D array of training targets of
            shape ``[B, 1]``, where ``B`` denotes the batch size. Targets
            determine the coloring of points in ``data_X``, values above
            ``<0.5`` are depicted in a different color than the remaining ones.
        samples_in (numpy.ndarray, optional): 2D array of sample inputs of shape
            ``[B, 2]``, where ``B`` denotes the batch size.
        filename (str, optional): If given, the image will be stored via the
            given filename.
        out_dir (str, optional): Directory where the images should be stored.
        ts (int): Text font size.
        lw (int): Line width.
        figsize (tuple): Figure size.
    """
    assert grid_X is not None and grid_Y is not None

    if isinstance(value_grid, torch.Tensor):
        value_grid = value_grid.detach().cpu().numpy()

    value_grid = value_grid.reshape(-1, 1)
    value_grid = value_grid.reshape(grid_X.shape)

    fig, axes = plt.subplots(figsize=figsize)
    if show_title and title is not None:
        plt.title(title, size=ts, pad=ts)

    extend = 'neither' # Colorbar extend.
    if vmin is None:
        vmin = value_grid.min()
    else:
        if crop:
            value_grid[value_grid < vmin] = vmin
        elif np.any(value_grid < vmin):
            extend = 'min'
    if vmax is None:
        vmax = value_grid.max()
    else:
        if crop:
            value_grid[value_grid > vmax] = vmax
        elif np.any(value_grid > vmax):
            extend = 'max' if extend == 'neither' else 'both'

    if vmax - vmin < 1e-5:
        vmax += 1e-5

    if force_extend is not None:
        extend = force_extend

    levels = None
    if log_levels:
        try:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), num_levels)
        except:
            log_levels = False
            warn('Cannot use logarithmic scale!')

    if levels is None:
        levels = np.linspace(vmin, vmax, num_levels)
    f = plt.contourf(grid_X, grid_Y, value_grid, cmap='coolwarm', levels=levels,
                     vmin=vmin, vmax=vmax, extend=extend)
    if cticks is None:
        cbar = plt.colorbar(f)
    else:
        assert ctick_lbls is not None
        cbar = plt.colorbar(f, ticks=cticks)
        cbar.ax.set_yticklabels(ctick_lbls)
    cbar.ax.tick_params(labelsize=ts, length=lw, width=lw/2.)

    if data_in is not None:
        colors = hmisc.get_colorbrewer2_colors(family='Dark2')
        plt.scatter(data_in[data_trgt<.5,0],
                    data_in[data_trgt<.5,1], c=colors[4])
        plt.scatter(data_in[data_trgt>=.5,0],
                    data_in[data_trgt>=.5,1], c=colors[5])

    if samples_in is not None:
        plt.scatter(samples_in[:, 0], samples_in[:, 1], c='k')

    axes.grid(False)
    axes.set_facecolor('w')
    axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
    axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
    if xticks is not None:
        plt.xticks(xticks, fontsize=ts)
    if yticks is not None:
        plt.yticks(yticks, fontsize=ts)
    axes.tick_params(axis='both', length=lw, direction='out', width=lw/2.)

    #plt.xlabel('$x_1$', fontsize=ts)
    #plt.ylabel('$x_2$', fontsize=ts)

    if filename is not None:
        if out_dir is None:
            out_dir = './'

        fpath = os.path.join(out_dir, filename)
        plt.savefig(fpath + '.pdf', bbox_inches='tight')
        plt.savefig(fpath + '.png', bbox_inches='tight')

    plt.show()

def plot_predictive_distributions_1dr(data, inputs, pd_samples=None,
                                      pd_mean=None, pd_std=None, sigma_ll=None,
                                      tf_writer=None, tf_tag=None, tf_step=None,
                                      title=None, show_title=True, xticks=None,
                                      yticks=None, xlim=None, ylim=None,
                                      vlines=None, show_legend=True,
                                      filename=None, out_dir=None,
                                      ts=30, lw=15, ms=5, figsize=(12, 7),
                                      show_plot=True):
    """Plot the predictive distribution for a 1D regression task.

    This plotting function is exclusively for instances of class
    :class:`hypnettorch.data.special.regression1d_data.ToyRegression`.

    Args:
        (....): See docstring of function :func:`heatmap`.
        data (hypnettorch.data.special.regression1d_data.ToyRegression): The
            underlying dataset object.
        inputs (numpy.ndarray): An array of x-values.
        pd_samples (numpy.ndarray, optional): An array of function draws
            (assuming each function encodes the mean of a Gaussian likelihood).
            The array is expected to be of shape ``[n_x, K]``, where ``[n_x]``
            is the shape of ``inputs``, and ``K`` denotes the number of
            functions to be plotted.
        pd_mean (numpy.ndarray, optional): An array of the same shape as
            ``inputs``, encoding the mean of the predictive posterior.
        pd_std (numpy.ndarray, optional): An array of the same shape as
            ``inputs``, encoding the std of the predictive posterior.
        sigma_ll (float, optional): The standard deviation of the Gaussian
            likelihood. Will only influence the plotting of function draws if
            ``pd_samples`` is provided.
        xlim (tuple, optional): The x-limits.
        ylim (tuple, optional): The y-limits.
        vlines (list, optional): A list of x-positions. For each entry, a
            vertical dotted line will be added.
        show_legend (bool): Whether the legend should be shown.
        tf_writer: Tensorboard summary writer.
        tf_tag (str): Tensorboard summary tag.
        tf_step (int): tensorboard summary global step.
        ms (int): Marker size.
        show_plots (bool): Whether the plot should be shown.
    """
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if isinstance(pd_samples, torch.Tensor):
        pd_samples = pd_samples.detach().cpu().numpy()
    if isinstance(pd_mean, torch.Tensor):
        pd_mean = pd_mean.detach().cpu().numpy()
    if isinstance(pd_std, torch.Tensor):
        pd_std = pd_std.detach().cpu().numpy()

    if len(inputs.shape) == 2:
        assert inputs.shape[1] == 1
        inputs = inputs.flatten()
    if pd_mean is not None and len(pd_mean.shape) == 2:
        assert pd_mean.shape[1] == 1
        pd_mean = pd_mean.flatten()
    if pd_std is not None and len(pd_std.shape) == 2:
        assert pd_std.shape[1] == 1
        pd_std = pd_std.flatten()

    #colors = ['#56641a', '#e6a176', '#00678a', '#b51d14', '#5eccab', '#3e474c',
    #          '#00783e']
    colors = hmisc.get_colorbrewer2_colors(family='Set2')

    fig, axes = plt.subplots(figsize=figsize)
    if show_title and title is not None:
        plt.title(title, size=ts, pad=ts)

    if data is not None:
        assert isinstance(data, ToyRegression)

        sample_x = np.linspace(start=inputs.min(), stop=inputs.max(),
                               num=100).reshape((-1, 1))
        sample_y = data._map(sample_x)

        plt.plot(sample_x, sample_y, color='k', linestyle='dashed',
                 linewidth=lw / 7, label='True Mean')

        train_x = data.get_train_inputs().squeeze()
        train_y = data.get_train_outputs().squeeze()

        plt.plot(train_x, train_y, 'o', color='k', label='Training Data',
                 markersize=ms)

    if pd_mean is not None:
        assert pd_std is not None

        cc = colors[1] # '#00678a'
        plt.plot(inputs, pd_mean, color=cc, label='Pred. dist.', lw=lw / 3.)

        plt.fill_between(inputs, pd_mean + pd_std, pd_mean - pd_std,
                         color=cc, alpha=0.3)
        plt.fill_between(inputs, pd_mean + 2. * pd_std, pd_mean - 2. * pd_std,
                         color=cc, alpha=0.2)
        plt.fill_between(inputs, pd_mean + 3. * pd_std, pd_mean - 3. * pd_std,
                         color=cc, alpha=0.1)

    if pd_samples is not None:
        K = pd_samples.shape[1]
        cc = cm.winter(np.linspace(0, 1, K))
        for i in range(K):
            lbl = 'Sample' if i == 0 else None
            plt.plot(inputs, pd_samples[:,i], alpha=0.9, c=cc[i], zorder=0,
                     label=lbl, lw=lw / 5.)
            if sigma_ll is not None:
                plt.fill_between(inputs, pd_samples[:,i] + sigma_ll,
                                 pd_samples[:,i] - sigma_ll, color=cc[i],
                                 alpha=0.1)

    if show_legend:
        plt.legend()

    #plt.xlabel('$x$', fontsize=ts)
    #plt.ylabel('$y$', fontsize=ts)

    axes.grid(False)
    axes.set_facecolor('w')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xticks is not None:
        plt.xticks(xticks, fontsize=ts)
    if yticks is not None:
        plt.yticks(yticks, fontsize=ts)
    axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
    axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
    if vlines is not None:
        for vx in vlines:
            axes.axvline(x=vx, color='k', lw=lw/6, ls=':')
    axes.tick_params(axis='both', length=lw, direction='out', width=lw/2.)

    if filename is not None:
        if out_dir is None:
            out_dir = './'

        fpath = os.path.join(out_dir, filename)
        plt.savefig(fpath + '.pdf', bbox_inches='tight')
        plt.savefig(fpath + '.png', bbox_inches='tight')

    if tf_writer is not None:
        tf_writer.add_figure(tf_tag, plt.gcf(), tf_step, close=not show_plot)

    if show_plot:
        hmisc.repair_canvas_and_show_fig(plt.gcf())
        plt.show()

    return fig


if __name__ == '__main__':
    pass


