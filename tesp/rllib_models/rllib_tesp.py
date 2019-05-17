# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from ray.rllib.models.lstm import add_time_dimension
from rllib_models.maml_model import MAMLModel
from keras_models.keras_tesp import KerasTESP


class RLlibTESP(MAMLModel):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        options = options["custom_options"]
        rnn_units = options["rnn_units"]
        rnn_output_units = options["rnn_output_units"]
        mlp_hidden_units = list(options["mlp_hidden_units"])
        vf_share_layers = options["vf_share_layers"]
        use_linear_baseline = options["linear_baseline"]
        rnn_output_activation = options.get("rnn_output_activation",
                                            "tanh")
        mlp_activation = options.get("mlp_activation",
                                     "tanh")

        with tf.name_scope("rnn_inputs"):
            rnn_inputs = tf.concat(
                [
                    input_dict["obs_buffer"],
                    input_dict["logits_buffer"],
                    tf.expand_dims(input_dict["reward_buffer"], axis=1)
                ], axis=1)
            with tf.name_scope("add_time_dimension"):
                rnn_inputs = add_time_dimension(rnn_inputs, self.seq_lens)
        mlp_inputs = input_dict["obs"]
        model_inputs = {
            "rnn": rnn_inputs,
            "seq_lens": self.seq_lens,
            "mlp": mlp_inputs}
        model = KerasTESP(
            rnn_units,
            rnn_output_units,
            rnn_output_activation,
            mlp_hidden_units + [num_outputs],
            mlp_activation,
            vf_share_layers=vf_share_layers,
            custom_params=self.custom_params["policy"],
            use_linear_baseline=use_linear_baseline)
        self.policy_model = model
        if use_linear_baseline:
            output = model(model_inputs)
        else:
            output, self._value_function = model(model_inputs)

        # last_layer = model.mlp.layers[-1]

        # self.rnn_state_in = self.keras_model.initial_state
        # if not nest.is_sequence(self.rnn_state_in):
        #     self.rnn_state_in = [self.rnn_state_in]
        self.rnn_state_out = nest.flatten(self.policy_model.state)
        self.rnn_state_out_init = [
            np.zeros(state.shape.as_list(), dtype=state.dtype.as_numpy_dtype)
            for state in self.rnn_state_out]

        return output, None

    def loss(self):
        reg_loss = tf.reduce_mean(tf.square(self.rnn_state_out))
        return reg_loss

    @staticmethod
    def prepare(policy_graph, options):
        rnn_input_dim = policy_graph.observation_space.shape[0] \
            + policy_graph.logit_dim + 1
        options = options["custom_options"]
        variables = {}
        with tf.name_scope("variables"):
            variables["policy"] = KerasTESP.get_variables(
                rnn_input_dim,
                options["rnn_units"],
                options["rnn_output_units"],
                options["projection_kernel_init_value"])
        return variables

    def extra_trainable_variables(self):
        weights = self.policy_model.mlp_policy.trainable_weights
        if hasattr(self.policy_model, "mlp_vf"):
            weights += self.policy_model.mlp_vf.trainable_weights
        return weights


if __name__ == "__main__":
    # import numpy as np
    # x = tf.constant(np.random.rand(10, 5).astype(np.float32))
    x = tf.placeholder(tf.float32, shape=(100, 5))
    input_dict = {
        "obs": x}
    obs_space = None
    num_outputs = 2
    options = {
        "rnn_units": 32,
        "rnn_output_units": 4,
        "mlp_hidden_units": [100, 100],
        "custom_options": {"vf_share_layers": True}
    }
    custom_variables = RLlibTESP.prepare(5, options)
    model = RLlibTESP(
        input_dict,
        obs_space,
        num_outputs,
        options,
        custom_params=custom_variables)

    writer = tf.summary.FileWriter(logdir="./summary",
                                   graph=tf.get_default_graph())
    writer.flush()
