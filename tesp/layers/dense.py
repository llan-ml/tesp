# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import layers as keras_layers


class Dense(keras_layers.Dense):
    @classmethod
    def get_variables(cls, input_units, output_units, **kwargs):
        dummy_instance = cls(output_units, **kwargs)
        with tf.name_scope(dummy_instance._name_scope()):
            dummy_instance.build(input_shape=(None, input_units))
        # maybe just return useful infos, e.g., regularization loss
        # instead of the dummy instance
        variables = {
            # "dummy": dummy_instance,
            "kernel": dummy_instance.kernel}
        if dummy_instance.use_bias:
            variables["bias"] = dummy_instance.bias
        return variables

    def __init__(self, units, custom_params=None, **kwargs):
        # assert (units is not None) != (custom_params is not None)
        # units = units or custom_params["kernel"].shape.as_list()[-1]
        keras_layers.Dense.__init__(self, units, **kwargs)

        if custom_params:
            assert isinstance(custom_params, dict)
            assert custom_params["kernel"].shape.as_list()[1] == units
            if self.use_bias:
                assert custom_params["bias"].shape.as_list()[0] == units
        self.custom_params = custom_params

    def build(self, input_shape):
        if self.custom_params:
            self.kernel = self.custom_params["kernel"]
            if self.use_bias:
                self.bias = self.custom_params["bias"]
            self.built = True
        else:
            keras_layers.Dense.build(self, input_shape)
