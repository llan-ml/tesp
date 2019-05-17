# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

from tensorflow.python.util import nest
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import _global_registry, RLLIB_MODEL

from maml.base_maml_policy_graph import BaseMAMLPolicyGraph
from losses import A3CLoss, PPOLoss
from baselines.linear_feature_baseline import LinearFeatureBaseline
from utils import compute_returns

logger = logging.getLogger("ray.rllib.agents.maml.maml_policy_graph")


class MAMLPolicyGraph(PPOPolicyGraph, BaseMAMLPolicyGraph):
    def __init__(self,
                 observation_space,
                 action_space,
                 config):
        # config = dict(ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, **config)
        self.sess = tf.get_default_session()
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.kl_coeff_val = self.config["kl_coeff"]
        self.kl_target = self.config["kl_target"]
        self.inner_lr = self.config["inner_lr"]
        self.outer_lr = self.config["outer_lr"]
        self.adaptive_inner_lr = self.config["adaptive_inner_lr"]
        # self.inner_lr_lower_bound = \
        #     self.config.get("inner_lr_lower_bound")
        #     # self.config.get("inner_lr_lower_bound") or - np.inf
        # self.inner_lr_upper_bound = \
        #     self.config.get("inner_lr_upper_bound")
        #     # self.config.get("inner_lr_upper_bound") or np.inf

        # if self.inner_lr == 0.01:
        #     if self.inner_lr_lower_bound is None:
        #         self.inner_lr_lower_bound = - 0.1
        #     self.inner_lr_upper_bound = 0.1
        # elif self.inner_lr == 0.001:
        #     if self.inner_lr_lower_bound is None:
        #         self.inner_lr_lower_bound = - 0.01
        #     self.inner_lr_upper_bound = 0.01
        # else:
        #     raise TypeError

        self.inner_lr_bound = self.config["inner_lr_bound"]
        if self.inner_lr_bound is None:
            self.inner_lr_lower_bound = - np.inf
            self.inner_lr_upper_bound = np.inf
        else:
            assert self.inner_lr_bound > 0
            self.inner_lr_lower_bound = - self.inner_lr_bound
            self.inner_lr_upper_bound = self.inner_lr_bound

        self.use_linear_baseline = \
            self.config["model"]["custom_options"]["linear_baseline"]
        self.mode = self.config["mode"]
        assert self.mode in ["local", "remote"]
        assert self.kl_coeff_val == 0.0

        dist_cls, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])

        self.logit_dim = logit_dim
        self.build_loss_in_phs()

        assert self.config["model"]["custom_model"]
        # logger.info(
        #     f'Using custom model {self.config["model"]["custom_model"]}')
        model_cls = _global_registry.get(RLLIB_MODEL,
                                         self.config["model"]["custom_model"])
        inner_variables = model_cls.prepare(self, self.config["model"])

        if self.adaptive_inner_lr:
            adaptive_learning_rates = model_cls.get_adaptive_learning_rates(
                inner_variables, self.inner_lr)

        def func(x, y):
            if self.adaptive_inner_lr:
                lr = tf.clip_by_value(
                    adaptive_learning_rates[x.op.name],
                    self.inner_lr_lower_bound,
                    self.inner_lr_upper_bound)
                return x - lr * y
            else:
                return x - self.inner_lr * y

        new_variables, grad_placeholders = \
            model_cls.build_placeholders_and_transform(
                inner_variables, func)

        self.model = model_cls(
            self.get_model_input_dict(),
            observation_space,
            logit_dim,
            self.config["model"],
            state_in=self.existing_state_in,
            seq_lens=self.existing_seq_lens,
            custom_params=new_variables)
        self.model_loss = self.model.loss()

        self._inner_variables = nest.flatten(inner_variables)
        self._outer_variables = nest.flatten(inner_variables)
        if hasattr(self.model, "extra_trainable_variables"):
            self._outer_variables += self.model.extra_trainable_variables()
        if self.adaptive_inner_lr:
            flat_adaptive_lrs = nest.flatten(adaptive_learning_rates)
            self._outer_variables.extend(flat_adaptive_lrs)
        self._outer_variable_dict = {
            var.op.name: var for var in self._outer_variables}

        self._grad_phs_loss_inputs = [
            (var.op.name, grad_ph)
            for var, grad_ph in zip(self._inner_variables,
                                    nest.flatten(grad_placeholders))]
        self._grad_phs_loss_input_dict = dict(self._grad_phs_loss_inputs)

        self.logits = self.model.outputs
        with tf.name_scope("sampler"):
            curr_action_dist = dist_cls(self.logits)
            self.sampler = curr_action_dist.sample()

        if self.config["use_gae"]:
            if self.use_linear_baseline:
                self.linear_baseline = LinearFeatureBaseline()
            self.value_function = self.model.value_function()
        else:
            raise NotImplementedError

        if self.model.state_in:
            raise NotImplementedError
        else:
            mask = None

        with tf.name_scope("a3c_loss"):
            self.a3c_loss_obj = A3CLoss(
                action_dist=curr_action_dist,
                actions=self.all_phs["actions"],
                advantages=self.all_phs["advantages"],
                value_targets=self.all_phs["value_targets"],
                vf_preds=self.all_phs["vf_preds"],
                value_function=self.value_function,
                vf_loss_coeff=self.config["vf_loss_coeff"],
                entropy_coeff=self.config["entropy_coeff"],
                vf_clip_param=self.config["vf_clip_param"])
            # self.a3c_loss_obj = PGLoss(
            #     curr_action_dist, act_ph, adv_ph)
        with tf.name_scope("ppo_loss"):                  # write own PPO loss, boolean_mask -> dynamic_partition
            self.ppo_loss_obj = PPOLoss(
                action_dist=curr_action_dist,
                action_space=action_space,
                logits=self.all_phs["logits"],
                actions=self.all_phs["actions"],
                advantages=self.all_phs["advantages"],
                value_targets=self.all_phs["value_targets"],
                vf_preds=self.all_phs["vf_preds"],
                value_function=self.value_function,
                valid_mask=mask,
                kl_coeff=self.kl_coeff_val,
                clip_param=self.config["clip_param"],
                vf_clip_param=self.config["vf_clip_param"],
                vf_loss_coeff=self.config["vf_loss_coeff"],
                entropy_coeff=self.config["entropy_coeff"],
                use_gae=self.config["use_gae"])

        self.total_loss = self.ppo_loss_obj.loss \
            + self.config["model_loss_coeff"] * self.model_loss

        BaseMAMLPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=self.all_phs["obs"],
            action_sampler=self.sampler,
            inner_loss=self.a3c_loss_obj.loss,
            inner_loss_inputs=self.a3c_loss_in,
            # outer_loss=self.ppo_loss_obj.loss,
            outer_loss=self.total_loss,
            outer_loss_inputs=self.ppo_loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            prev_action_input=self.all_phs["prev_actions"],
            prev_reward_input=self.all_phs["prev_rewards"],
            seq_lens=self.model.seq_lens,
            max_seq_len=self.config["model"]["max_seq_len"])

        self.a3c_stats_fetches = {
            "total_loss": self.a3c_loss_obj.loss,
            "policy_loss": self.a3c_loss_obj.mean_policy_loss,
            "vf_loss": self.a3c_loss_obj.mean_vf_loss,
            "entropy": self.a3c_loss_obj.mean_entropy
        }
        self.ppo_stats_fetches = {
            "total_loss": self.total_loss,
            "ppo_loss": self.ppo_loss_obj.loss,
            "policy_loss": self.ppo_loss_obj.mean_policy_loss,
            "vf_loss": self.ppo_loss_obj.mean_vf_loss,
            "entropy": self.ppo_loss_obj.mean_entropy,
            "kl": self.ppo_loss_obj.mean_kl,
            "model_loss": self.model_loss
        }

        self.sess.run(tf.global_variables_initializer())
        # self.clear_grad_buffer()

    def build_loss_in_phs(self):
        with tf.name_scope("inputs"):
            obs_ph = tf.placeholder(
                tf.float32,
                shape=(None, ) + self.observation_space.shape,
                name="obs")
            adv_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="advantages")
            act_ph = ModelCatalog.get_action_placeholder(self.action_space)
            logits_ph = tf.placeholder(
                tf.float32, shape=(None, self.logit_dim), name="logits")
            vf_preds_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="vf_preds")
            value_targets_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="value_targets")
            prev_actions_ph = \
                ModelCatalog.get_action_placeholder(self.action_space)
            prev_rewards_ph = tf.placeholder(
                tf.float32, shape=(None, ), name="prev_rewards")
            self.existing_state_in = None
            self.existing_seq_lens = None
            extra_phs = self.build_extra_phs()
        self.a3c_loss_in = [
            ("obs", obs_ph),
            ("advantages", adv_ph),
            ("actions", act_ph),
            ("value_targets", value_targets_ph),
            ("vf_preds", vf_preds_ph),
            ("prev_actions", prev_actions_ph),
            ("prev_rewards", prev_rewards_ph)
        ]
        self.ppo_loss_in = list(self.a3c_loss_in) \
            + [("logits", logits_ph)]
        self.all_phs = {
            "obs": obs_ph,
            "advantages": adv_ph,
            "actions": act_ph,
            "value_targets": value_targets_ph,
            "vf_preds": vf_preds_ph,
            "prev_actions": prev_actions_ph,
            "prev_rewards": prev_rewards_ph,
            "logits": logits_ph}
        self.all_phs.update(extra_phs)

    def build_extra_phs(self):
        return {}

    def get_model_input_dict(self):
        return {
            "obs": self.all_phs["obs"],
            "prev_actions": self.all_phs["prev_actions"],
            "prev_rewards": self.all_phs["prev_rewards"]}

    def reset(self):
        self.clear_grad_buffer()

    def clear_grad_buffer(self):
        self._grad_buffer = {
            name: np.zeros(ph.shape.as_list(),
                           dtype=ph.dtype.as_numpy_dtype)
            for name, ph in self._grad_phs_loss_input_dict.items()}

    def update_grad_buffer(self, grad_values):
        for key, grad in grad_values.items():
            self._grad_buffer[key] += grad

    def get_inner_grad_feed_dict(self):
        feed_dict = {
            self._grad_phs_loss_input_dict[name]: self._grad_buffer[name]
            for name in self._grad_phs_loss_input_dict}
        return feed_dict

    def extra_compute_action_feed_dict(self):
        return self.get_inner_grad_feed_dict()

    def extra_compute_action_fetches(self):
        if self.use_linear_baseline:
            return {"logits": self.logits}
        else:
            return PPOPolicyGraph.extra_compute_action_fetches(self)

    def extra_compute_grad_feed_dict(self):
        return self.get_inner_grad_feed_dict()

    def extra_compute_grad_fetches(self):
        return self.stats_fetches

    def _get_inner_grads(self):
        inner_grads = \
            tf.gradients(self._inner_loss, self._inner_variables,
                         name="inner_gradients")
        if self.config["inner_grad_clip"]:
            with tf.name_scope("inner_grad_clip"):
                inner_grads, _ = tf.clip_by_global_norm(
                    inner_grads, self.config["inner_grad_clip"])
        return {
            v.op.name: g
            for v, g in zip(self._inner_variables, inner_grads)
            if g is not None}

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.config["outer_lr"])

    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        if self.use_linear_baseline:
            # fit and predict after sampling all trajectories
            batch = compute_returns(
                sample_batch,
                0.0,
                self.config["gamma"])
            return batch
        else:
            return PPOPolicyGraph.postprocess_trajectory(
                self, sample_batch, other_agent_batches, episode)


if __name__ == "__main__":
    import gym
    import ray
    from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
    from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
    from ray.tune.logger import pretty_print
    from fcnet import FullyConnectedNetwork

    # ray.init()
    ModelCatalog.register_custom_model("maml_mlp", FullyConnectedNetwork)

    config = {
        "inner_lr": 0.5,
        "outer_lr": 0.0001,
        "use_gae": True,
        "vf_share_layers": True,
        "horizon": 200,
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "custom_model": "maml_mlp",
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "max_seq_len": 20,
            "custom_options": {"vf_share_layers": True}
        }
    }
    config = dict(DEFAULT_CONFIG, **config)
    print(pretty_print(config))

    sess = tf.InteractiveSession()

    def env_creator(config):
        return gym.make("CartPole-v1")

    evaluator = PolicyEvaluator(
        env_creator,
        MAMLPolicyGraph,
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
        log_level=config["log_level"])
    policy = evaluator.policy_map["default"]
    batch = evaluator.sample()
    grads, infos = policy.compute_inner_gradients(batch)

    # observation_space = env.observation_space
    # action_space = env.action_space
    # policy_graph = MAMLPolicyGraph(observation_space, action_space, config)
    # graph = tf.get_default_graph()
    # writer = tf.summary.FileWriter(logdir="./summary", graph=graph)
    writer = tf.summary.FileWriter(logdir="./summary", graph=evaluator.tf_sess.graph)
    writer.flush()
