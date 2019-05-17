# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models.misc import get_activation_fn

from rllib_models.maml_model import MAMLModel
from keras_models.keras_mlp import KerasMLP


class RLlibMLP(MAMLModel):
    def _build_layers(self, inputs, num_outputs, options):
        options = options["custom_options"]
        mlp_hidden_units = options["mlp_hidden_units"]
        vf_share_layers = options["vf_share_layers"]
        use_linear_baseline = options["linear_baseline"]
        mlp_activation = options.get("mlp_activation",
                                     "tanh")
        # assert mlp_activation == "relu"
        model = KerasMLP(
            layer_units_exclude_first=mlp_hidden_units + [num_outputs],
            activation=mlp_activation,
            custom_params=self.custom_params.get("policy"),
            vf_share_layers=vf_share_layers)
        self.policy_model = model
        if vf_share_layers:
            output, self._value_function = model(inputs)
        else:
            output = model(inputs)
            if not use_linear_baseline:
                self.vf_model = KerasMLP(
                    # layer_units_exclude_first=mlp_hidden_units[:1] + [1],
                    layer_units_exclude_first=mlp_hidden_units + [1],
                    activation=mlp_activation,
                    custom_params=self.custom_params["vf"],
                    name="keras_mlp_vf")
                self._value_function = self.vf_model(inputs)
        last_layer = model.layers[-1]
        return output, last_layer

    @staticmethod
    def prepare(policy_graph, options):
        custom_options = options["custom_options"]
        observation_space = policy_graph.observation_space
        output_dim = policy_graph.logit_dim // 2 \
            if options["free_log_std"] else policy_graph.logit_dim
        assert len(observation_space.shape) == 1
        all_units = observation_space.shape \
            + tuple(custom_options["mlp_hidden_units"]) \
            + (output_dim, )
        vf_share_layers = custom_options["vf_share_layers"]
        variables = {}
        with tf.name_scope("variables"):
            variables["policy"] = KerasMLP.get_variables(
                all_units, vf_share_layers)
            if (not custom_options["linear_baseline"]) \
                    and (not custom_options["vf_share_layers"]):
                variables["vf"] = KerasMLP.get_variables(
                    # all_units[:2] + (1, ), name="keras_mlp_vf")
                    all_units[:-1] + (1, ), name="keras_mlp_vf")
        return variables


if __name__ == "__main__":
    import os
    tf.set_random_seed(1)
    mode = 2

    x = tf.placeholder(tf.float32, [None, 10], "x")
    # x = keras_layers.Input(tensor=x)

    input_dict = {"obs": x, "prev_action": None, "prev_reward": None}
    obs_space = None
    num_outputs = 5
    options = {
        "fcnet_hiddens": [20],
        "fcnet_activation": "tanh",
        "free_log_std": False,
        "vf_share_layers": True}

    if mode == 1:
        model = FullyConnectedNetwork(
            input_dict=input_dict,
            obs_space=obs_space,
            num_outputs=num_outputs,
            options=options)
    elif mode == 2:
        new_variables, placeholders, custom_variables, dummy_variables = \
            FullyConnectedNetwork.prepare([10, 20, 5], vf_share_layers=True)
        model = FullyConnectedNetwork(
            input_dict,
            obs_space,
            num_outputs,
            options,
            custom_params=new_variables)

    init_op = tf.global_variables_initializer()

    graph = tf.get_default_graph()
    os.system("mkdir -p summary")
    writer = tf.summary.FileWriter(logdir="./summary", graph=graph)
    writer.flush()

    sess = tf.Session()
    sess.run(init_op)
    # print(sess.run(tf.trainable_variables()))
