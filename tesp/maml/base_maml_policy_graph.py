# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph
from ray.rllib.utils.tf_run_builder import TFRunBuilder


class BaseMAMLPolicyGraph(TFPolicyGraph):
    def __init__(self,
                 observation_space,
                 action_space,
                 sess,
                 obs_input,
                 action_sampler,
                 inner_loss,
                 inner_loss_inputs,
                 outer_loss,
                 outer_loss_inputs,
                 state_inputs=None,
                 state_outputs=None,
                 prev_action_input=None,
                 prev_reward_input=None,
                 seq_lens=None,
                 max_seq_len=20):
        self.observation_space = observation_space
        self.action_space = action_space
        self._sess = sess
        self._obs_input = obs_input
        self._prev_action_input = prev_action_input
        self._prev_reward_input = prev_reward_input
        self._sampler = action_sampler
        self._inner_loss = inner_loss
        self._inner_loss_inputs = inner_loss_inputs
        self._inner_loss_input_dict = dict(self._inner_loss_inputs)
        self._outer_loss = outer_loss
        self._outer_loss_inputs = outer_loss_inputs
        self._outer_loss_input_dict = dict(self._outer_loss_inputs)
        self._is_training = tf.placeholder_with_default(True, ())
        self._state_inputs = state_inputs or []
        self._state_outputs = state_outputs or []
        for i, ph in enumerate(self._state_inputs):
            for d in [self._inner_loss_input_dict,
                      self._outer_loss_input_dict]:
                d[f"state_in_{i}"] = ph
        self._seq_lens = seq_lens
        self._max_seq_len = max_seq_len

        # if self.mode == "remote":
        self._inner_grads = self._get_inner_grads()

        self._outer_grads = \
            tf.gradients(self._outer_loss, self._outer_variables,
                         name="outer_gradients")
        self._outer_grads = {
            v.op.name: g
            for v, g in zip(self._outer_variables, self._outer_grads)
            if g is not None}

        if self.mode == "local":
            with tf.name_scope("outer_optimizer"):
                self._optimizer = self.optimizer()
                grads_and_vars = [
                    (self._outer_grads[name], self._outer_variable_dict[name])
                    for name in self._outer_grads]
                self._apply_op = self._optimizer.apply_gradients(grads_and_vars)

        if len(self._state_inputs) != len(self._state_outputs):
            raise ValueError(
                "Number of state input and output tensors must match, got: "
                "{} vs {}".format(self._state_inputs, self._state_outputs))
        if len(self.get_initial_state()) != len(self._state_inputs):
            raise ValueError(
                "Length of initial state must match number of state inputs, "
                "got: {} vs {}".format(self.get_initial_state(),
                                       self._state_inputs))
        if self._state_inputs and self._seq_lens is None:
            raise ValueError(
                "seq_lens tensor must be given if state inputs are defined")

    def _before_compute_grads(self):
        for attr in ["_grads", "_loss_inputs", "_loss_input_dict",
                     "stats_fetches"]:
            if hasattr(self, attr):
                raise TypeError

    def _after_compute_grads(self):
        for attr in ["_grads", "_loss_inputs", "_loss_input_dict",
                     "stats_fetches"]:
            delattr(self, attr)

    def compute_inner_gradients(self, postprocessed_batch):
        builder = TFRunBuilder(self._sess, "compute_inner_gradients")
        self._before_compute_grads()
        self._grads = self._inner_grads
        self._loss_inputs = self._inner_loss_inputs
        self._loss_input_dict = self._inner_loss_input_dict
        self.stats_fetches = self.a3c_stats_fetches
        fetches = self.build_compute_gradients(builder, postprocessed_batch)
        results = builder.get(fetches)
        self._after_compute_grads()
        return results

    def compute_outer_gradients(self, postprocessed_batch):
        builder = TFRunBuilder(self._sess, "compute_outer_gradients")
        self._before_compute_grads()
        self._grads = self._outer_grads
        self._loss_inputs = self._outer_loss_inputs
        self._loss_input_dict = self._outer_loss_input_dict
        self.stats_fetches = self.ppo_stats_fetches
        fetches = self.build_compute_gradients(builder, postprocessed_batch)
        results = builder.get(fetches)
        self._after_compute_grads()
        return results

    def build_apply_gradients(self, builder, gradients):
        assert len(gradients) == len(self._outer_grads), \
            (gradients, self._outer_grads)
        builder.add_feed_dict(self.extra_apply_grad_feed_dict())
        builder.add_feed_dict({self._is_training: True})
        builder.add_feed_dict({
            self._outer_grads[name]: gradients[name]
            for name in self._outer_grads})
        fetches = builder.add_fetches(
            [self._apply_op, self.extra_apply_grad_fetches()])
        return fetches[1]

    def get_weights(self):
        return self._sess.run(self._outer_variable_dict)

    def set_weights(self, weights):
        assert isinstance(weights, dict)
        for name, var in self._outer_variable_dict.items():
            var.load(weights[name], session=self._sess)
