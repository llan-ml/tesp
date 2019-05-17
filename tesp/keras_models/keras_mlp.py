# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models as keras_models

from layers.dense import Dense
from keras_models.misc import normc_initializer


class KerasMLP(keras_models.Model):
    @classmethod
    def get_variables(cls, all_units,
                      vf_share_layers=False, name=None):
        name = name or "keras_mlp_policy"
        if "policy" in name:
            last_kernel_init_value = 0.01
        elif "vf" in name:
            last_kernel_init_value = 1.0
        else:
            raise NotImplementedError
        variables = {}
        with tf.name_scope(name):
            for i, (size_in, size_out) in \
                    enumerate(zip(all_units, all_units[1:])):
                name = f"dense_{i}"
                variables[name] = \
                    Dense.get_variables(
                        size_in,
                        size_out,
                        name=name,
                        kernel_initializer=(
                            normc_initializer(1.0) if i < len(all_units) - 2
                            else normc_initializer(last_kernel_init_value)))
            if vf_share_layers:
                name = f"dense_vf"
                variables[name] = \
                    Dense.get_variables(
                        size_in,
                        1,
                        name=name,
                        kernel_initializer=normc_initializer(1.0))
        # tricky to remove the name count of the dummy instance
        # since the default name is used here
        # graph = tf.get_default_graph()
        # K.PER_GRAPH_LAYER_NAME_UIDS[graph][
        #     ("", dummy_instance.name)] -= 1
        return variables

    # @staticmethod
    # def filter_dummy_variables(variables):
    #     assert isinstance(variables, dict)
    #     filtered_variables = {}
    #     for name, var in variables.items():
    #         tmp = {}
    #         for key, value in var.items():
    #             if key in ["kernel", "bias"]:
    #                 tmp[key] = value
    #         filtered_variables[name] = tmp
    #     return filtered_variables

    def __init__(self, layer_units_exclude_first, activation,
                 custom_params=None, vf_share_layers=False, name=None):
        """
            layer_units: list, a list of the number of units of all layers
                except the input layer
        """
        name = name or "keras_mlp_policy"
        if "policy" in name:
            last_kernel_init_value = 0.01
        elif "vf" in name:
            last_kernel_init_value = 0.01
        else:
            raise NotImplementedError
        keras_models.Model.__init__(self, name=name)

        custom_params = custom_params or {}
        for i, size in enumerate(layer_units_exclude_first):
            name = f"dense_{i}"
            layer = Dense(
                size,
                custom_params=custom_params.get(name),
                activation=(
                    activation if i < len(layer_units_exclude_first) - 1
                    else None),
                kernel_initializer=(
                    normc_initializer(1.0)
                    if i < len(layer_units_exclude_first) - 1
                    else normc_initializer(last_kernel_init_value)),
                name=name)
            setattr(self, name, layer)
        if vf_share_layers:
            name = f"dense_vf"
            layer = Dense(
                1,
                custom_params=custom_params.get(name),
                activation=None,
                kernel_initializer=normc_initializer(1.0),
                name=name)
            setattr(self, name, layer)
        self._vf_share_layers = vf_share_layers

    def call(self, inputs):
        last_inputs = inputs
        last_shared_index = -2 if self._vf_share_layers else -1
        for layer in self.layers[:last_shared_index]:
            last_inputs = layer(last_inputs)
        output = self.layers[last_shared_index](last_inputs)
        if self._vf_share_layers:
            value_function = self.layers[-1](last_inputs)
            return output, value_function
        else:
            return output
