# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as keras_layers


class GRUCell(keras_layers.GRUCell):
    @classmethod
    def get_variables(cls, input_units, output_units, **kwargs):
        dummy_instance = cls(output_units, **kwargs)
        with tf.name_scope(dummy_instance._name_scope()):
            dummy_instance.build(input_shape=(None, input_units))
        variables = {
            # "dummy": dummy_instance,
            "kernel": dummy_instance.kernel,
            "recurrent_kernel": dummy_instance.recurrent_kernel}
        if dummy_instance.use_bias:
            variables["bias"] = dummy_instance.bias
        return variables

    def __init__(self, units, custom_params=None, **kwargs):
        # assert (units is not None) != (custom_params is not None)
        # units = units or custom_params["recurrent_kernel"].shape.as_list()[0]
        keras_layers.GRUCell.__init__(self, units, **kwargs)

        if custom_params:
            assert isinstance(custom_params, dict)
            assert custom_params["kernel"].shape.as_list()[1] == 3 * units \
                and custom_params[
                    "recurrent_kernel"].shape.as_list() == [units, 3 * units]
            if self.use_bias:
                assert custom_params["bias"].shape.as_list()[0] == 3 * units
        self.custom_params = custom_params

    def build(self, input_shape):
        if self.custom_params:
            self.kernel = self.custom_params["kernel"]
            self.recurrent_kernel = self.custom_params["recurrent_kernel"]
            if self.use_bias:
                self.bias = self.custom_params["bias"]
                if not self.reset_after:
                    self.input_bias, self.recurrent_bias = self.bias, None
                else:
                    self.input_bias = K.flatten(self.bias[0])
                    self.recurrent_bias = K.flatten(self.bias[1])
            else:
                self.bias = None
            self.built = True
        else:
            keras_layers.GRUCell.build(self, input_shape)
