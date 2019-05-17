# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from ray.rllib.evaluation.sample_batch import MultiAgentBatch, \
    DEFAULT_POLICY_ID

from maml.maml_policy_evaluator import MAMLPolicyEvaluator

logger = logging.getLogger("ray.rllib.agents.tesp.tesp_policy_evaluator")
# logger.setLevel(logging.DEBUG)


class TESPPolicyEvaluator(MAMLPolicyEvaluator):
    def _inner_update_once(self, warmup=False):
        policy = self.policy_map[DEFAULT_POLICY_ID]
        if warmup:
            samples = self.sample()
            policy.update_episode_buffer(samples)
        samples = self.sample()
        if isinstance(samples, MultiAgentBatch):
            raise NotImplementedError
        else:
            inner_grads, _ = policy.compute_inner_gradients(samples)
            policy.update_grad_buffer(inner_grads)
            policy.update_episode_buffer(samples)
            # inner_infos["batch_count"] = samples.count
        return samples

    def inner_update(self, num_inner_updates):
        policy = self.policy_map[DEFAULT_POLICY_ID]
        policy.reset()
        self.episodes = {}
        self.post_samples = None
        goals = []
        for i in range(num_inner_updates):
            samples = self._inner_update_once(i == 0)
            self.episodes[str(i)] = self.sampler.get_metrics()
            goals.append(self._get_goal(samples))
        self.post_samples = self.sample()
        goals.append(self._get_goal(self.post_samples))
        assert np.allclose(np.mean(goals, axis=0), goals[0])
        # logger.debug(f"goal: {goals[0]}")
        self.episodes[str(num_inner_updates)] = self.sampler.get_metrics()
        return goals[0]
