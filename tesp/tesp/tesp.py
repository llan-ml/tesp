# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ray.rllib.utils import merge_dicts

from maml.maml import MAMLAgent, DEFAULT_CONFIG
from tesp.tesp_policy_graph import TESPPolicyGraph
from tesp.tesp_policy_evaluator import TESPPolicyEvaluator


logger = logging.getLogger("ray.rllib.agents.tesp.tesp")

DEFAULT_CONFIG = merge_dicts(
    DEFAULT_CONFIG,
    {
        "adaptive_inner_lr": True,
        "episode_mode": "episode_buffer",
        "buffer_size": 16,
        "model": {
            "custom_model": "RLlibTESP",
            "custom_options": {
                "rnn_units": 64,
                "rnn_output_units": 4,
                "mlp_hidden_units": [128, 128],
                "projection_kernel_init_value": 0.01,
                "vf_share_layers": False,
                "linear_baseline": True
            }
        }
    }
)


class TESPAgent(MAMLAgent):
    _agent_name = "TESP"
    _default_config = DEFAULT_CONFIG
    _policy_graph = TESPPolicyGraph
    _policy_evaluator = TESPPolicyEvaluator

    def _validate_config(self):
        # if self.config["inner_lr"]:
        #     assert self.config["adaptive_inner_lr"]
        pass


if __name__ == "__main__":
    import time
    import ray
    import numpy as np
    import tensorflow as tf
    from ray.tune.registry import register_env
    from ray.rllib.models.catalog import ModelCatalog
    from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
    from ray.rllib.evaluation.metrics import summarize_episodes
    from ray.tune.logger import pretty_print
    from rllib_models.rllib_tesp import RLlibTESP
    from envs.point_env import PointEnv
    from envs.mujoco.ant import AntEnv
    from envs.reset_wrapper import ResetWrapper

    logger = logging.getLogger("ray.rllib.agents")
    logger.setLevel(logging.DEBUG)

    # ray.init()
    ray.init(redis_address="192.168.12.39:32222")
    env_cls = AntEnv
    model_cls = RLlibTESP
    register_env(env_cls.__name__,
                 lambda env_config: ResetWrapper(env_cls(), env_config))
    # register_env("PointEnv", lambda env_config: PointEnv(env_config))
    ModelCatalog.register_custom_model(model_cls.__name__, model_cls)

    config = {
        # "num_workers": 20,
        "model": {
            "custom_model": model_cls.__name__,
            "custom_options": {
                "rnn_units": 256,
                "rnn_output_units": 16,
                "mlp_hidden_units": [512, 512],
                "vf_share_layers": False,
                "linear_baseline": True}
            # "squash_to_range": True,
            # "free_log_std": True
        }
    }

    agent = TESPAgent(config=config, env=env_cls.__name__)
    evaluator = agent.local_evaluator
    policy = evaluator.policy_map[DEFAULT_POLICY_ID]
    optimizer = agent.optimizer

    # for i in range(10):
    #     st = time.time()
    #     logger.info(f"\n{i}")
    #     res = agent.train()
    #     logger.info(f'\n{pretty_print(res["inner_update_metrics"])}')

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

    # writer = tf.summary.FileWriter(logdir="./summary", graph=evaluator.tf_sess.graph)
    # writer.flush()
