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
# @title          :finite_width/rbf_net.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :06/28/2021
# @version        :1.0
# @python_version :3.8.10
r"""
Multi-Layer RBF Network
-----------------------

The module :mod:`finite_width.rbf_net` implements a multi-layer RBF network,
where output activations of a layer are computed as

.. math::

    \mathbf{x}^l = W^l \mathbf{z}^l + \mathbf{b}^l

with

.. math::

    \mathbf{z}^l = h(\mathbf{x}^{l-1}, \mathbf{u}^l) = \
        \exp \bigg( - \frac{1}{2 \sigma_g^2} \
        \lVert \mathbf{x}^{l-1} - \mathbf{u}^l  \rVert_2^2 \bigg)

Further information can be found in module :mod:`nngp.rbf_net`. Note, while in
module :mod:`nngp.rbf_net` layer-widths tend to infinity
:math:`N_l \rightarrow \infty`, here :math:`N_l` has to be finite and an
embedding vector :math:`\mathbf{u}^l` is learned for each of the :math:`N_l`
units in a layer in conjunction with the linear weights :math:`W^l` and
:math:`\mathbf{b}^l`.

Note, this kind of multi-layer RBF network can be considered as multiple RBF
networks stacked on top of each other.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypnettorch.mnets.mnet_interface import MainNetInterface

class StackedRBFNet(nn.Module, MainNetInterface):
    """Implementation of a Multi-Layer RBF network.

    Args:
        n_in (int): Number of inputs.
        n_nonlin_units (list or tuple): Number :math:`N_l` of non-linear units
            :math:`\mathbf{z}^l` for each layer :math:`l`.
        n_lin_units (list or tuple): Number :math:`M_l` of linear units
            :math:`\mathbf{x}^l` for each layer :math:`l`. The last entry in
            this list will determine the number of outputs.

            Note:
                ``n_lin_units`` and ``n_nonlin_units`` must have the same
                length, which determines the number of layers.
        use_bias (bool): Whether layers may have bias terms.
        bandwidth (float): The bandwidth parameter :math:`\sigma_g^2`.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        verbose (bool): Whether to print information (e.g., the number of
            weights) during the construction of the network.
    """
    def __init__(self, n_in=1, n_nonlin_units=(10,), n_lin_units=(1,),
                 use_bias=True, bandwidth=1., no_weights=False, verbose=True):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        if len(n_nonlin_units) != len(n_lin_units):
            raise ValueError('Arguments "n_nonlin_units" and "n_lin_units" ' +
                             'must have the same length!')
        if len(n_nonlin_units) == 0:
            raise ValueError('Network needs to have at least 1 layer!')

        ### Setup class attributes ###
        self._n_nonlinear = [None] + list(n_nonlin_units)
        self._n_linear = [n_in] + list(n_lin_units)
        self._n_layers = len(n_lin_units)
        self._bandwidth = bandwidth

        self._has_bias = use_bias
        self._no_weights = no_weights

        # The output `x^L` is always obtained via a linear transformation from
        # `z^L`.
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer!
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = None if no_weights else nn.ParameterList()
        self._hyper_shapes_learned = None if not no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        ### Instantiate layers ###
        lin_in_sizes = []
        lin_out_sizes = []
        layer_inds = []

        for l in range(1, len(self._n_linear)):
            N_l = self._n_nonlinear[l]
            M_l_prev = self._n_linear[l-1]
            M_l = self._n_linear[l]

            # Create embeddings `u_l`.
            s = [N_l, M_l_prev]

            if not no_weights:
                self._internal_params.append(torch.nn.Parameter( \
                    torch.Tensor(*s), requires_grad=True))
                # TODO Better way of initializing `u^l`?
                torch.nn.init.normal_(self._internal_params[-1], mean=0.,
                                      std=1.)
            else:
                self._hyper_shapes_learned.append(s)
                self._hyper_shapes_learned_ref.append( \
                    len(self.param_shapes))

            self._param_shapes.append(s)
            self._param_shapes_meta.append({
                'name': 'embedding', # Is `u^l` really an embedding.
                'index': -1 if no_weights else len(self._internal_params)-1,
                'layer': l
            })

            # The linear layer that maps z^l to x^l is constructed below.
            lin_in_sizes.append(N_l)
            lin_out_sizes.append(M_l)
            layer_inds.append(l)

        self._add_fc_layers(lin_in_sizes, lin_out_sizes, self._no_weights,
                            fc_layers=layer_inds)

        ###########################
        ### Print infos to user ###
        ###########################
        if verbose:
            print('Creating a "%s" with %d weights' \
                  % (str(self), self.num_params))

        self._is_properly_setup()

    def __str__(self):
        return '%d-layer RBF network' % (self._n_layers)

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`hypnettorch.mnets.mnet_interface.MainNetInterface.\
distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`hypnettorch.mnets.mnet_interface.MainNetInterface.\
forward`.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if self._no_weights and weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        if distilled_params is not None:
            raise ValueError('Argument "distilled_params" has no influence ' +
                             'on this method')

        if condition is not None:
            raise ValueError('Argument "condition" has no influence on this ' +
                             'method')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        if weights is None:
            weights = self.weights

        # Disentangle `u^l`, `W^l` and `b^l`
        ind = 0
        u_weights = weights[:self._n_layers]
        ind += self._n_layers

        w_weights = []
        b_weights = []

        for l in range(self._n_layers):
            w_weights.append(weights[ind])
            ind += 1
            if self.has_bias:
                b_weights.append(weights[ind])
                ind += 1
            else:
                b_weights.append(None)

        ###########################
        ### Forward Computation ###
        ###########################

        h = x

        for l in range(self._n_layers):
            # Now, `h` encodes `x^{l-1}`.
            # Compute z^l
            h = ((h[:, None, :] - u_weights[l][None, :, :])**2).sum(dim=2)
            h = torch.exp(- 1 / (2 * self._bandwidth) * h)

            # Compute x^l
            h = F.linear(h, w_weights[l], bias=b_weights[l])

        return h

if __name__ == '__main__':
    pass


