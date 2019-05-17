# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import defaultdict

import ray
import numpy as np
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer

from utils import summarize_episodes

logger = logging.getLogger("ray.rllib.agents.maml.maml_optimizer")


class MAMLOptimizer(PolicyOptimizer):
    def _init(self, num_inner_updates=1, num_sgd_iter=1):
        self.num_inner_updates = num_inner_updates
        self.num_sgd_iter = num_sgd_iter

    def sync_weights(self):
        if self.remote_evaluators:
            weights = self.local_evaluator.get_weights()
            weights_id = ray.put(weights)
            # import ipdb; ipdb.set_trace()
            # for e in self.remote_evaluators:
            #     e.set_weights.remote(weights_id)
            ray.get([e.set_weights.remote(weights_id)
                     for e in self.remote_evaluators])
            return weights
        else:
            raise TypeError

    def step(self):
        # distribute weights of the model and the outer optimizer
        prev_weights = self.sync_weights()

        # perform the inner update in each remote evaluator
        goals = ray.get([
            e.inner_update.remote(self.num_inner_updates)
            for e in self.remote_evaluators])
        assert isinstance(goals, list)
        goals = sorted(goals, key=lambda x: (x[0], x[1]))
        logger.debug(f"\ngoals:\n{np.asarray(goals)}")

        best_policy_loss = None
        best_infos = None
        best_weights = None
        last_weights = None
        kl_target = self.local_evaluator.policy_config["kl_target"]

        # gather the gradients and update the variables in the local evaluator
        for i in range(self.num_sgd_iter):
            dist_outer_grad_values, dist_outer_infos = zip(
                *ray.get([e.outer_update.remote()
                          for e in self.remote_evaluators]))

            aggregated_grads = defaultdict(list)
            aggregated_infos = defaultdict(list)
            for outer_grad_values, outer_infos \
                    in zip(dist_outer_grad_values, dist_outer_infos):
                for name, values in outer_grad_values.items():
                    aggregated_grads[name].append(values)
                for name, infos in outer_infos.items():
                    aggregated_infos[name].append(infos)
            aggregated_grads = dict(aggregated_grads)
            aggregated_infos = dict(aggregated_infos)

            grad_values = {k: np.mean(v, axis=0)
                           for k, v in aggregated_grads.items()}
            infos = {k: np.mean(v) for k, v in aggregated_infos.items()}
            logger.debug(f"\niter: {i}, infos: {infos}")

            if infos["kl"] < kl_target:
                if best_policy_loss is None \
                        or infos["policy_loss"] < best_policy_loss:
                    best_policy_loss = infos["policy_loss"]
                    best_infos = infos
                    best_weights = last_weights or prev_weights

            if i < self.num_sgd_iter - 1:
                fetches = self.local_evaluator.apply_gradients(grad_values)
                assert fetches == {}
                last_weights = self.sync_weights()

        self.local_evaluator.set_weights(best_weights)
        self.sync_weights()

        return best_infos

    def collect_metrics(self):
        dist_episodes = ray.get([
            e.apply.remote(lambda ev: ev.episodes)
            for e in self.remote_evaluators])

        aggregated_episodes = defaultdict(list)
        for episodes in dist_episodes:
            for k, v in episodes.items():
                aggregated_episodes[k].extend(v)
        aggregated_episodes = dict(aggregated_episodes)

        res = {k: summarize_episodes(v, v, 0)
               for k, v in aggregated_episodes.items()}

        return {"inner_update_metrics": res}
                # "reward_attr": res[
                #     str(self.num_inner_updates)]["episode_reward_mean"]}
