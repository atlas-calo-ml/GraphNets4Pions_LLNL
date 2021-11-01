# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model architectures for the demos in TensorFlow 2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
from six.moves import range
import sonnet as snt
import tensorflow as tf


NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.

_EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": True,
    }

_NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": False,
    "use_nodes": True,
    "use_globals": True,
    }

_GLOBAL_BLOCK_OPT = {
    "use_edges": False,
    "use_nodes": True,
    "use_globals": True,
    }

# def make_mlp_model(activate_final=True):
#     """Instantiates a new MLP, followed by LayerNorm.

#     The parameters of each new MLP are not shared with others generated by
#     this function.

#     Returns:
#     A Sonnet module which contains the MLP and LayerNorm.
#     """
#     return snt.Sequential([
#         snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=activate_final),
#         snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
#     ])

def make_mlp_model(latent_size, num_layers, activate_final=True):
        """Instantiates a new MLP, followed by LayerNorm.

        The parameters of each new MLP are not shared with others generated by
        this function.

        Returns:
        A Sonnet module which contains the MLP and LayerNorm.
        """
        return snt.Sequential([
            snt.nets.MLP([latent_size] * num_layers, activate_final=activate_final),
            snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])


class DSBlockModel(snt.Module):
    """

    """

    def __init__(self,
               node_output_size=None,
               global_output_size=None,
               num_blocks=3,
               num_layers=2,
               latent_size=16,
               concat_input=True,
               name="DSBlockModel"):
        super(DSBlockModel, self).__init__(name=name)

        self._num_blocks = num_blocks
        self._num_layers = num_layers
        self._latent_size = latent_size
        self._concat_input = concat_input

        self._core = [
                modules.DeepSets(
                        node_model_fn=lambda: make_mlp_model(self._latent_size, self._num_layers),
                        global_model_fn=lambda: make_mlp_model(self._latent_size, self._num_layers),
                        reducer=tf.math.unsorted_segment_sum,
                        name="core_"+str(i)
                    ) for i in range(self._num_blocks)
            ]


        # Transforms the outputs into the appropriate shapes.
        edge_fn = None
        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, name="node_output")
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")

        self._output_transform = modules.GraphIndependent(
            edge_fn, node_fn, global_fn, name="network_output")

    def __call__(self, input_op):
        latent = self._core[0](input_op)
        latent_all = [input_op]
        for i in range(1, self._num_blocks):
            if self._concat_input:
                core_input = utils_tf.concat([latent, latent_all[-1]], axis=1)
            else:
                core_input = latent

            latent_all.append(latent)
            latent = self._core[i](core_input)

        latent_all.append(latent)
        stacked_latent = utils_tf.concat(latent_all, axis=1)
        output = (self._output_transform(stacked_latent))
        return output