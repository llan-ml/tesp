# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
import logging
from collections import defaultdict
import pickle
import numpy as np
import ray
from ray.rllib.agents import Agent
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as ppo_default_config
from ray.rllib.utils import merge_dicts
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.env.env_context import EnvContext
from ray import tune
from ray.tune.trial import Resources

from maml.maml_policy_graph import MAMLPolicyGraph
from maml.maml_optimizer import MAMLOptimizer
from maml.maml_policy_evaluator import MAMLPolicyEvaluator
from envs.reset_wrapper import ResetArgsHolder
from utils import summarize_episodes

logger = logging.getLogger("ray.rllib.agents.maml.maml")


def on_episode_start(info):
    episode = info["episode"]
    episode.custom_metrics["episode_dist_reward"] = 0


def on_episode_step(info):
    episode = info["episode"]
    episode.custom_metrics["episode_dist_reward"] += \
        episode.batch_builder.agent_builders[
            "single_agent"].buffers["infos"][-1]["dist_reward"]


DEFAULT_CONFIG = merge_dicts(
    ppo_default_config,
    {
        "random_seed": 1,
        "inner_lr": 0.05,
        "outer_lr": 1e-3,
        "adaptive_inner_lr": False,
        # "inner_lr_lower_bound": None,
        # "inner_lr_upper_bound": None,
        "inner_lr_bound": None,
        "num_inner_updates": 3,
        "inner_grad_clip": 40.0,
        "num_tasks": 100,
        "clip_param": 0.2,
        # "linear_baseline": False,
        "use_gae": True,

        # "gamma": 0.99,
        # "lambda": 0.97,
        "gamma": 0.0,
        "lambda": 0.0,
        "horizon": 200,
        "kl_coeff": 0.0,
        "kl_target": 0.01,
        "entropy_coeff": 0.0,
        "vf_loss_coeff": 0.05,
        "vf_clip_param": 15.0,
        "model_loss_coeff": 0.0,

        "num_sgd_iter": 10,
        "validation": True,
        "validation_freq": 5,
        "sample_batch_size": 200,
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "num_workers": 20,
        "num_envs_per_worker": 25,
        "tf_session_args": {
            "intra_op_parallelism_threads": 4,
            "inter_op_parallelism_threads": 4
        },

        "callbacks": {
            "on_episode_start": tune.function(on_episode_start),
            "on_episode_step": tune.function(on_episode_step)
        }
    }
)


class MAMLAgent(Agent):
    _agent_name = "MAML"
    _default_config = DEFAULT_CONFIG
    _policy_graph = MAMLPolicyGraph
    _policy_evaluator = MAMLPolicyEvaluator

    @classmethod
    def default_resource_request(cls, config):
        cf = merge_dicts(cls._default_config, config)
        return Resources(
            cpu=1,
            gpu=0,
            extra_cpu=cf["num_cpus_per_worker"] * cf["num_workers"] + 1,
            extra_gpu=cf["num_gpus_per_worker"] * cf["num_workers"])

    def make_local_evaluator(self, env_creator, policy_dict):
        return self._make_evaluator(
            self._policy_evaluator,
            env_creator,
            policy_dict,
            0,
            merge_dicts(
                self.config, {
                    "tf_session_args": {
                        "intra_op_parallelism_threads": 8,
                        "inter_op_parallelism_threads": 8
                    }
                }
            ))

    def make_remote_evaluators(self, env_creator, policy_dict, count,
                               remote_args):
        cls = self._policy_evaluator.as_remote(**remote_args).remote
        return [
            self._make_evaluator(cls, env_creator, policy_dict, i + 1,
                                 self.config) for i in range(count)
        ]

    def _init(self):
        self._validate_config()
        env = self.env_creator(env_config={})

        reset_args_shape = (env.reset_args_config["shape"][0], )
        self.reset_args_holder = ResetArgsHolder.remote(
            (self.config["num_workers"], ) + reset_args_shape)
        self.config["env_config"] = merge_dicts(
            self.config["env_config"],
            {"reset_args_holder": self.reset_args_holder})

        self.rng = np.random.RandomState(self.config["random_seed"])
        # print("sampling goals...")
        self.reset_args_train, self.reset_args_test_1, self.reset_args_test_2 \
            = env.sample_reset_args(self.rng, self.config["num_tasks"])
        # print("sampling finished")
        self.reset_args_test = {
            1: self.reset_args_test_1,
            2: self.reset_args_test_2}

        observation_space = env.observation_space
        action_space = env.action_space
        policy_dict_local = {
            DEFAULT_POLICY_ID: (
                self._policy_graph,
                observation_space,
                action_space,
                {"mode": "local"})}
        policy_dict_remote = {
            DEFAULT_POLICY_ID: (
                self._policy_graph,
                observation_space,
                action_space,
                {"mode": "remote"})}

        self.local_evaluator = self.make_local_evaluator(
            self.env_creator, policy_dict_local)
        self.remote_evaluators = self.make_remote_evaluators(
            self.env_creator, policy_dict_remote, self.config["num_workers"], {
                "num_cpus": self.config["num_cpus_per_worker"],
                "num_gpus": self.config["num_gpus_per_worker"]})
        self.optimizer = MAMLOptimizer(
            self.local_evaluator, self.remote_evaluators, {
                "num_inner_updates": self.config["num_inner_updates"],
                "num_sgd_iter": self.config["num_sgd_iter"]})

    def _validate_config(self):
        assert not self.config["adaptive_inner_lr"]

    def _train(self):
        batch_reset_args_indices = \
            self.rng.choice(self.reset_args_train.shape[0],
                            size=self.config["num_workers"],
                            replace=False)
        batch_reset_args = self.reset_args_train[batch_reset_args_indices]
        ray.get(self.reset_args_holder.set.remote(batch_reset_args))

        fetches = self.optimizer.step()
        # if "kl" in fetches:
        #     raise NotImplementedError
        res = self.optimizer.collect_metrics()
        res.update(
            info=dict(fetches, **res.get("info", {})))

        res.update({"validation": None})
        if self.config["validation"]:
            if self.config["validation_freq"] == "auto":
                if self._iteration <= 200:
                    validation_freq = 100
                else:
                    validation_freq = 25
            else:
                validation_freq = self.config["validation_freq"]
            if (self._iteration + 1) % validation_freq == 0:
                val_results = self._validation_once()
                res.update({"validation": val_results})
        return res

    def _validation_once(self):
        val_results = {}
        val_results["train"] = self._test(
            self.reset_args_train, self.config["num_inner_updates"])
        val_results["test_1"] = self._test(
            self.reset_args_test_1, self.config["num_inner_updates"])
        val_results["test_2"] = self._test(
            self.reset_args_test_2, self.config["num_inner_updates"])
        return val_results

    def train(self):
        results = Agent.__base__.train(self)
        return results

    def _test(self, reset_args, num_inner_updates):
        num_tasks = reset_args.shape[0]
        free_evaluators = list(
            zip(range(1, self.config["num_workers"] + 1),
                self.remote_evaluators))
        weights_id = ray.put(self.local_evaluator.get_weights())
        running = {}
        finished = {}
        episodes = defaultdict(list)
        i = 0
        while True:
            if free_evaluators and i < num_tasks:
                this_index, this_evaluator = free_evaluators.pop()
                this_reset_args = reset_args[i]
                reset_args_holder_content = ray.get(self.reset_args_holder.get.remote())
                reset_args_holder_content = np.copy(reset_args_holder_content)
                reset_args_holder_content[this_index - 1] = this_reset_args
                ray.get(self.reset_args_holder.set.remote(reset_args_holder_content))
                ray.get(this_evaluator.set_weights.remote(weights_id))
                remote = this_evaluator.inner_update.remote(num_inner_updates)
                running[remote] = (
                    i, this_reset_args, this_index, this_evaluator)
                i += 1
                continue
            if running:
                ready_ids, _ = ray.wait(list(running.keys()),
                                        num_returns=1,
                                        timeout=1000)
                if ready_ids:
                    assert len(ready_ids) == 1
                    ready_id = ready_ids[0]
                    task_id, this_reset_args, this_index, this_evaluator = \
                        running.pop(ready_id)
                    assert np.array_equal(ray.get(ready_id), this_reset_args)
                    assert np.array_equal(reset_args[task_id], this_reset_args)
                    this_episodes = ray.get(
                        this_evaluator.apply.remote(lambda e: e.episodes))
                    for k, v in this_episodes.items():
                        episodes[k].extend(v)
                    finished[task_id] = (this_reset_args, this_episodes)
                    free_evaluators.append((this_index, this_evaluator))
                continue
            break
        return {k: summarize_episodes(v, v, 0) for k, v in episodes.items()}

    def _stop(self):
        self.reset_args_holder.__ray_terminate__.remote()
        Agent._stop(self)

    def _save(self, checkpoint_dir):
        checkpoint_path = osp.join(checkpoint_dir,
                                   f"checkpoint-{self.iteration}")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(self.__getstate__(), f)
            pickle.dump({"reset_args_train": self.reset_args_train,
                         "reset_args_test": self.reset_args_test}, f)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            extra_data = pickle.load(f)
            reset_args = pickle.load(f)
        self.__setstate__(extra_data)
        self.reset_args_train = reset_args["reset_args_train"]
        self.reset_args_test = reset_args["reset_args_test"]


if __name__ == "__main__":
    import time
    import ray
    import numpy as np
    import tensorflow as tf
    from ray.tune.registry import register_env
    from ray.rllib.models.catalog import ModelCatalog
    from ray.rllib.evaluation.metrics import summarize_episodes
    from ray.tune.logger import pretty_print
    from rllib_models.rllib_mlp import RLlibMLP
    from envs.point_env import PointEnv
    from envs.mujoco.reacher import ReacherEnv
    from envs.reset_wrapper import ResetWrapper

    logger = logging.getLogger("ray.rllib.agents.maml")
    logger.setLevel(logging.DEBUG)

    ray.init()
    # ray.init(redis_address="localhost:32222")
    env_cls = PointEnv
    env_cls = ReacherEnv
    register_env(env_cls.__name__,
                 lambda env_config: ResetWrapper(env_cls(env_config), env_config))
    # register_env("PointEnv", lambda env_config: PointEnv(env_config))
    ModelCatalog.register_custom_model("maml_mlp", RLlibMLP)

    config = {
        "env_config": {"ctrl_cost_coeff": 0.0},
        # "num_workers": 20,
        "model": {
            "custom_model": "maml_mlp",
            "fcnet_hiddens": [100, 100],
            "fcnet_activation": "tanh",
            "custom_options": {"vf_share_layers": True},
            # "squash_to_range": True,
            # "free_log_std": True
        }
    }

    agent = MAMLAgent(config=config, env=env_cls.__name__)
    evaluator = agent.local_evaluator
    policy = evaluator.policy_map[DEFAULT_POLICY_ID]
    optimizer = agent.optimizer

    for i in range(10):
        st = time.time()
        logger.info(f"\n{i}")
        res = agent.train()
        logger.info(f'\n{pretty_print(res["inner_update_metrics"])}')

    # only perform inner update in the local evaluator
    # policy.clear_grad_buffer()
    # def func():
    #     grads, infos, samples = evaluator._inner_update_once()
    #     policy.update_grad_buffer(grads)
    #     episodes = evaluator.sampler.get_metrics()
    #     logger.info(
    #         f'\n{pretty_print(summarize_episodes(episodes, episodes))}')
    #     logger.info(f"\n{pretty_print(infos)}")
    #     return grads, samples
    # for i in range(1000):
    #     print(i)
    #     grads, samples = func()

    writer = tf.summary.FileWriter(logdir="./summary", graph=evaluator.tf_sess.graph)
    writer.flush()
