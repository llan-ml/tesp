# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.tf_run_builder import TFRunBuilder
from ray.rllib.models.lstm import chop_into_sequences
from ray.rllib.evaluation.sample_batch import SampleBatch

from maml.maml_policy_graph import MAMLPolicyGraph
from utils import separate_sample_batch

logger = logging.getLogger("ray.rllib.agents.tesp.tesp_policy_graph")


class TESPPolicyGraph(MAMLPolicyGraph):
    def build_extra_phs(self):
        with tf.name_scope("buffer"):
            obs_buffer_ph = tf.placeholder(
                tf.float32,
                shape=(None, ) + self.observation_space.shape,
                name="obs")
            logits_buffer_ph = tf.placeholder(
                tf.float32,
                shape=(None, self.logit_dim),
                name="logits")
            reward_buffer_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="reward")
            seq_lens_buffer = tf.placeholder(
                tf.int32, shape=[None], name="seq_lens")
            self.existing_seq_lens = seq_lens_buffer
        return {
            "obs_buffer": obs_buffer_ph,
            "logits_buffer": logits_buffer_ph,
            "reward_buffer": reward_buffer_ph,
            "seq_lens_buffer": seq_lens_buffer}

    def get_model_input_dict(self):
        return {
            "obs": self.all_phs["obs"],
            "obs_buffer": self.all_phs["obs_buffer"],
            "logits_buffer": self.all_phs["logits_buffer"],
            "reward_buffer": self.all_phs["reward_buffer"]}

    def reset(self):
        MAMLPolicyGraph.reset(self)
        self.clear_episode_buffer()

    def clear_episode_buffer(self):
        self._episode_buffer = None
        self._rnn_state_out = None

    def update_episode_buffer(self, samples):
        if self.config["episode_mode"] == "episode_buffer":
            # assert True
            new_batches = list(separate_sample_batch(samples).values())
            if self._episode_buffer:
                old_batches = \
                    list(separate_sample_batch(self._episode_buffer).values())
            else:
                old_batches = []
            all_batches = old_batches + new_batches
            all_batches = sorted(
                all_batches, key=lambda x: x["rewards"].sum(), reverse=True)
            buffer_size = self.config["buffer_size"]
            # assert buffer_size == 24
            self._episode_buffer = SampleBatch.concat_samples(all_batches[:buffer_size])
        elif self.config["episode_mode"] == "last_episodes":
            # assert False
            self._episode_buffer = samples
        elif self.config["episode_mode"] == "all_episodes":
            # assert False
            if self._episode_buffer:
                self._episode_buffer = self._episode_buffer.concat(samples)
            else:
                self._episode_buffer = samples
        else:
            raise NotImplementedError
        self._rnn_state_out = None

    def _get_rnn_state_out_feed_dict(self):
        feature_keys = ["obs", "logits", "rewards"]
        features_sequences, _, seq_lens = chop_into_sequences(
            self._episode_buffer["eps_id"],
            [self._episode_buffer[k] for k in feature_keys],
            [],
            9999)
        feed_dict = {
            self.all_phs["obs_buffer"]: features_sequences[0],
            self.all_phs["logits_buffer"]: features_sequences[1],
            self.all_phs["reward_buffer"]: features_sequences[2],
            self.all_phs["seq_lens_buffer"]: seq_lens}
        return feed_dict

    def extra_compute_action_feed_dict(self):
        feed_dict = MAMLPolicyGraph.extra_compute_action_feed_dict(self)
        feed_dict.update({
            state: value
            for state, value in zip(self.model.rnn_state_out,
                                    self.compute_rnn_state_out())})
        return feed_dict

    def extra_compute_grad_feed_dict(self):
        feed_dict = MAMLPolicyGraph.extra_compute_grad_feed_dict(self)
        feed_dict.update(self._get_rnn_state_out_feed_dict())
        return feed_dict

    def build_compute_rnn_state_out(self, builder):
        builder.add_feed_dict(self.get_inner_grad_feed_dict())
        builder.add_feed_dict(self._get_rnn_state_out_feed_dict())
        fetches = builder.add_fetches(self.model.rnn_state_out)
        return fetches

    def compute_rnn_state_out(self):
        if self._episode_buffer:
            if not self._rnn_state_out:
                builder = TFRunBuilder(self._sess, "compute_rnn_state_out")
                fetches = self.build_compute_rnn_state_out(builder)
                self._rnn_state_out = builder.get(fetches)
            return self._rnn_state_out
        else:
            return self.model.rnn_state_out_init

    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        trajsize = len(sample_batch["actions"])
        for i, value in enumerate(self.compute_rnn_state_out()):
            sample_batch[f"rnn_state_out_{i}"] = [value] * trajsize
        return MAMLPolicyGraph.postprocess_trajectory(
            self, sample_batch, other_agent_batches, episode)


if __name__ == "__main__":
    import gym
    import ray
    from maml.maml import DEFAULT_CONFIG
    from tesp.tesp_policy_evaluator import TESPPolicyEvaluator
    from ray.rllib.utils import merge_dicts
    from ray.tune.logger import pretty_print
    from rllib_models.rllib_tesp import RLlibTESP

    # ray.init()
    ModelCatalog.register_custom_model("tesp", RLlibTESP)

    config = {
        "inner_lr": 0.5,
        "outer_lr": 0.0001,
        "use_gae": True,
        "vf_share_layers": True,
        "horizon": 200,
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "mode": "local",
        "model": {
            "custom_model": "tesp",
            "custom_options": {
                "rnn_units": 64,
                "rnn_output_units": 8,
                "mlp_hidden_units": [128, 128],
                "vf_share_layers": True}
        }
    }
    config = merge_dicts(DEFAULT_CONFIG, config)
    print(pretty_print(config))

    sess = tf.InteractiveSession()

    def env_creator(config):
        return gym.make("MountainCarContinuous-v0")

    evaluator = TESPPolicyEvaluator(
        env_creator,
        TESPPolicyGraph,
        batch_steps=config["sample_batch_size"],
        batch_mode=config["batch_mode"],
        episode_horizon=config["horizon"],
        preprocessor_pref=config["preprocessor_pref"],
        sample_async=config["sample_async"],
        compress_observations=config["compress_observations"],
        num_envs=config["num_envs_per_worker"],
        observation_filter=config["observation_filter"],
        clip_rewards=config["clip_rewards"],
        env_config=config["env_config"],
        model_config=config["model"],
        policy_config=config,
        worker_index=0,
        monitor_path=self.logdir if config["monitor"] else None,
        log_level=config["log_level"],
        callbacks=config["callbacks"])
    policy = evaluator.policy_map["default"]
    policy.clear_grad_buffer()
    policy.clear_episode_buffer()
    batch = evaluator.sample()
    policy.update_episode_buffer(batch)
    batch = evaluator.sample()
    grads, infos = policy.compute_inner_gradients(batch)
    grads, infos = policy.compute_outer_gradients(batch)

    # observation_space = env.observation_space
    # action_space = env.action_space
    # policy_graph = MAMLPolicyGraph(observation_space, action_space, config)
    # graph = tf.get_default_graph()
    # writer = tf.summary.FileWriter(logdir="./summary", graph=graph)
    writer = tf.summary.FileWriter(logdir="./summary", graph=evaluator.tf_sess.graph)
    writer.flush()
