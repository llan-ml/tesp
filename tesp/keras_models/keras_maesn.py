# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models as keras_models

from layers.dense import Dense
from keras_models.keras_mlp import KerasMLP
from keras_models.misc import normc_initializer


class KerasMAESN(keras_models.Model):
    @classmethod
    def get_variables(cls,
                      latent_units,
                      projection_kernel_init_value,
                      # mlp_input_units,
                      # mlp_units,
                      # vf_share_layers=False
                      name=None):
        name = name or "keras_maesn_policy"
        variables = {}
        with tf.name_scope(name):
            variables["latent_z"] = tf.Variable(
                initial_value=normc_initializer(projection_kernel_init_value)(shape=(latent_units, )),
                trainable=True,
                dtype=tf.float32,
                name="latent_z")
            # variables["cell"] = GRUCell.get_variables(
            #     rnn_input_units, rnn_units)
            # variables["projection"] = Dense.get_variables(
            #     rnn_units,
            #     rnn_output_units,
            #     name="dense_projection",
            #     kernel_initializer=normc_initializer(projection_kernel_init_value))
            # variables["mlp"] = KerasMLP.get_dummy_variables(
            #     [mlp_input_units + rnn_output_units] + mlp_units,
            #     vf_share_layers=vf_share_layers, **mlp_kwargs)

        # graph = tf.get_default_graph()
        # K.PER_GRAPH_LAYER_NAME_UIDS[graph][
        #     ("", dummy_instance.name)] -= 1
        return variables

    def __init__(self,
                 latent_units,
                 mlp_units_exclude_first,
                 mlp_activation,
                 custom_params=None,
                 vf_share_layers=False,
                 use_linear_baseline=False):
        keras_models.Model.__init__(self, name="keras_maesn_policy")

        custom_params = custom_params or {}
        self.latent_z = custom_params.get("latent_z")
        assert latent_units == self.latent_z.shape.as_list()[0]
        self.mlp_policy = KerasMLP(
            layer_units_exclude_first=mlp_units_exclude_first,
            activation=mlp_activation,
            vf_share_layers=vf_share_layers,
            name="keras_mlp_policy")
        if not vf_share_layers and not use_linear_baseline:
            self.mlp_vf = KerasMLP(
                layer_units_exclude_first=mlp_units_exclude_first[:-1] + [1],
                activation=mlp_activation,
                vf_share_layers=False,
                name="keras_mlp_vf")

    def call(self, inputs):
        assert isinstance(inputs, dict)
        # assert inputs["rnn"].shape.ndims == 3 \
        #     and inputs["mlp"].shape.ndims == 2 \
        #     and inputs["seq_lens"].shape.ndims == 1
        # rnn_inputs = inputs["rnn"]
        mlp_inputs = inputs["mlp"]
        # seq_lens = inputs["seq_lens"]
        # if getattr(self, "initial_state", None) is None:
        #     with tf.name_scope("initial_state"):
        #         self.initial_state = self.cell.get_initial_state(rnn_inputs)
        # outputs, state = tf.nn.dynamic_rnn(
        #     cell=self.cell,
        #     inputs=rnn_inputs,
        #     sequence_length=seq_lens,
        #     initial_state=self.initial_state,
        #     dtype=rnn_inputs.dtype)
        # projected_state = self.dense_projection(state)
        # with tf.name_scope("average"):
        #     self.state = tf.reduce_mean(projected_state, axis=0, keepdims=True)
        with tf.name_scope("latent_z"):
            self.state = tf.expand_dims(self.latent_z, axis=0)
        with tf.name_scope("mlp_inputs"):
            tiled_state = tf.manip.tile(self.state,
                                        multiples=[tf.shape(mlp_inputs)[0], 1])
            mlp_inputs = tf.concat([mlp_inputs, tiled_state], axis=1)
        if self.mlp_policy._vf_share_layers:
            outputs, value_function = self.mlp_policy(mlp_inputs)
            return outputs, value_function
        else:
            outputs = self.mlp_policy(mlp_inputs)
            if hasattr(self, "mlp_vf"):
                value_function = self.mlp_vf(mlp_inputs)
                return outputs, value_function
            else:
                return outputs


if __name__ == "__main__":
    rnn_inputs = tf.placeholder(tf.float32, shape=[None, None, 5])
    mlp_inputs = tf.placeholder(tf.float32, shape=[None, 2])
    seq_lens = tf.placeholder(tf.int64, shape=[None])
    # model = KerasTESP(32,
    #                   4,
    #                   "tanh",
    #                   [100, 100, 2],
    #                   "tanh",
    #                   vf_share_layers=True)
    # outputs, value_function = model({"rnn": rnn_inputs,
    #                                  "seq_lens": seq_lens,
    #                                  "mlp": mlp_inputs})

    with tf.name_scope("variables"):
        custom_variables = KerasTESP.get_variables(
            5, 32, 4)
    model = KerasTESP(
        32,
        4,
        "tanh",
        [100, 100, 2],
        "tanh",
        custom_params=custom_variables,
        vf_share_layers=True)
    outputs, value_function = model({"rnn": rnn_inputs,
                                     "seq_lens": seq_lens,
                                     "mlp": mlp_inputs})

    init_op = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init_op)

    writer = tf.summary.FileWriter(logdir="./summary",
                                   graph=tf.get_default_graph())
    writer.flush()
