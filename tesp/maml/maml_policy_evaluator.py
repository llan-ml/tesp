# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf
from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
from ray.rllib.evaluation.sample_batch import MultiAgentBatch, \
    DEFAULT_POLICY_ID
from ray.rllib.env.async_vector_env import _VectorEnvToAsync
from ray.rllib.evaluation.sampler import SyncSampler, _env_runner
from ray.rllib.evaluation.metrics import summarize_episodes

from envs.reset_wrapper import ResetWrapper
from utils import postprocess_trajectory

logger = logging.getLogger("ray.rllib.agents.maml.maml_policy_evaluator")
# logger.setLevel(logging.DEBUG)


class MAMLPolicyEvaluator(PolicyEvaluator):
    def __init__(self,
                 env_creator,
                 policy_graph,
                 policy_mapping_fn=None,
                 policies_to_train=None,
                 tf_session_creator=None,
                 batch_steps=100,
                 batch_mode="truncate_episodes",
                 episode_horizon=None,
                 preprocessor_pref="deepmind",
                 sample_async=False,
                 compress_observations=False,
                 num_envs=1,
                 observation_filter="NoFilter",
                 clip_rewards=None,
                 env_config=None,
                 model_config=None,
                 policy_config=None,
                 worker_index=0,
                 monitor_path=None,
                 log_level=None,
                 callbacks=None):
        policy_config.pop("env_config")
        tf.set_random_seed(policy_config["random_seed"])
        PolicyEvaluator.__init__(self,
                                 env_creator,
                                 policy_graph,
                                 policy_mapping_fn,
                                 policies_to_train,
                                 tf_session_creator,
                                 batch_steps,
                                 batch_mode,
                                 episode_horizon,
                                 preprocessor_pref,
                                 sample_async,
                                 compress_observations,
                                 num_envs,
                                 observation_filter,
                                 clip_rewards,
                                 env_config,
                                 model_config,
                                 policy_config,
                                 worker_index,
                                 monitor_path,
                                 log_level,
                                 callbacks)

    def reset_sample(self):
        async_env = self.async_env
        sampler = self.sampler
        batch_mode = self.batch_mode
        if not isinstance(async_env, _VectorEnvToAsync) \
                or not isinstance(sampler, SyncSampler) \
                or batch_mode != "complete_episodes":
            raise NotImplementedError

        # reset async_env
        for env in async_env.vector_env.envs:
            try:
                while not isinstance(env, ResetWrapper):
                    env = env.env
                setattr(env, "with_reset_args", False)
            except AttributeError as error:
                logger.error(error)
        async_env.new_obs = async_env.vector_env.vector_reset()
        async_env.cur_rewards = [None for _ in range(async_env.num_envs)]
        async_env.cur_dones = [False for _ in range(async_env.num_envs)]
        async_env.cur_infos = [None for _ in range(async_env.num_envs)]

        # reset sampler
        sampler.async_vector_env = async_env
        sampler.rollout_provider = _env_runner(
            sampler.async_vector_env, sampler.extra_batches.put,
            sampler.policies, sampler.policy_mapping_fn,
            sampler.unroll_length, sampler.horizon,
            sampler._obs_filters, False, False, self.callbacks, self.tf_sess)
        sampler.get_metrics()
        sampler.get_extra_batches()

    def sample(self):
        self.reset_sample()
        samples = PolicyEvaluator.sample(self)
        policy = self.policy_map[DEFAULT_POLICY_ID]
        if policy.use_linear_baseline:
            samples = postprocess_trajectory(
                samples,
                policy.linear_baseline,
                self.policy_config["gamma"],
                self.policy_config["lambda"],
                self.policy_config["use_gae"])
        return samples

    def _inner_update_once(self):
        samples = self.sample()
        if isinstance(samples, MultiAgentBatch):
            raise NotImplementedError
        else:
            inner_grads, inner_infos = \
                self.policy_map[
                    DEFAULT_POLICY_ID].compute_inner_gradients(samples)
            inner_infos["batch_count"] = samples.count
        return inner_grads, inner_infos, samples

    def inner_update(self, num_inner_updates):
        policy = self.policy_map[DEFAULT_POLICY_ID]
        policy.reset()
        self.episodes = {}
        self.post_samples = None
        goals = []
        for i in range(num_inner_updates):
            inner_grad_values, inner_infos, samples = self._inner_update_once()
            policy.update_grad_buffer(inner_grad_values)
            self.episodes[str(i)] = self.sampler.get_metrics()
            goals.append(self._get_goal(samples))
        self.post_samples = self.sample()
        goals.append(self._get_goal(self.post_samples))
        assert np.allclose(np.mean(goals, axis=0), goals[0])
        # logger.debug(f"goal: {goals[0]}")
        self.episodes[str(num_inner_updates)] = self.sampler.get_metrics()
        return goals[0]

    def collect_metrics(self):
        assert self.episodes
        metrics = {k: summarize_episodes(v, v, 0)
                   for k, v in self.episodes.items()}
        return metrics

    def _get_goal(self, samples):
        infos = samples["infos"]
        goals = [info["goal"] for info in infos]
        assert np.allclose(np.mean(goals, axis=0), goals[0])
        return goals[0]

    def outer_update(self):
        assert hasattr(self, "post_samples") and self.post_samples is not None
        outer_grad_values, outer_infos = \
            self.policy_map[
                DEFAULT_POLICY_ID].compute_outer_gradients(self.post_samples)
        return outer_grad_values, outer_infos

    def apply_gradients(self, grads):
        return self.policy_map[DEFAULT_POLICY_ID].apply_gradients(grads)

    def set_weights(self, weights):
        for pid, w in weights.items():
            self.policy_map[pid].set_weights(w)
        return weights

