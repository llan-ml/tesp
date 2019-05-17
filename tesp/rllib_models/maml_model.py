# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.util import nest
from ray.rllib.models import model as rllib_model


class MAMLModel(rllib_model.Model):
    def __init__(self,
                 input_dict,
                 obs_space,
                 num_outputs,
                 options,
                 state_in=None,
                 seq_lens=None,
                 custom_params=None):
        self.custom_params = custom_params or {}
        rllib_model.Model.__init__(
            self,
            input_dict,
            obs_space,
            num_outputs,
            options,
            state_in,
            seq_lens)

    def value_function(self):
        if hasattr(self, "_value_function"):
            with tf.name_scope("flat_vf"):
                self._value_function = tf.reshape(self._value_function, [-1])
        else:
            self._value_function = None
        return self._value_function

    def loss(self):
        return tf.constant(0.0)

    @staticmethod
    def prepare(observation_space, output_dim, options, func=None):
        raise NotImplementedError

    @staticmethod
    def build_placeholders_and_transform(variables, func):
        flat_variables = nest.flatten(variables)
        assert all([isinstance(var, tf.Variable) for var in flat_variables])
        with tf.name_scope("placeholders"):
            flat_placeholders = [
                tf.placeholder(var.dtype, var.shape, _get_name(var.op.name))
                for var in flat_variables]
        with tf.name_scope("inner_sgd_step"):
            flat_new_variables = list(map(func,
                                          flat_variables, flat_placeholders))
        placeholders = nest.pack_sequence_as(variables, flat_placeholders)
        new_variables = nest.pack_sequence_as(variables, flat_new_variables)
        return new_variables, placeholders

    @staticmethod
    def get_adaptive_learning_rates(variables, init_value):
        flat_variables = nest.flatten(variables)
        assert all([isinstance(var, tf.Variable) for var in flat_variables])
        lrs = {}
        with tf.name_scope("adaptive_learning_rates"):
            lrs = {
                var.op.name:
                tf.Variable(
                    initial_value=tf.constant_initializer(init_value)(
                        var.shape, dtype=var.dtype),
                    dtype=var.dtype,
                    name=_get_name(var.op.name),
                    trainable=True)
                for var in flat_variables}
        return lrs


def _get_name(x):
    return re.sub(".*variables/", "", x)
