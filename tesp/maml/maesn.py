# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ray.rllib.utils import merge_dicts

from maml.maml import MAMLAgent, DEFAULT_CONFIG

logger = logging.getLogger("ray.rllib.agents.maml.maesn")

DEFAULT_CONFIG = merge_dicts(
    DEFAULT_CONFIG,
    {
        "adaptive_inner_lr": True,
        "model": {
            "custom_model": "RLlibMAESN",
            "custom_options": {
                "rnn_output_units": 4,
                "mlp_hidden_units": [64, 64],
                "projection_kernel_init_value": 0.01,
                "vf_share_layers": False,
                "linear_baseline": True
            }
        }
    }
)


class MAESNAgent(MAMLAgent):
    _agent_name = "MAESN"
    _default_config = DEFAULT_CONFIG

    def _validate_config(self):
        assert self.config["adaptive_inner_lr"]


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
    # from models.rllib_maesn
    from envs.point_env import PointEnv
    from envs.reset_wrapper import ResetWrapper

    logger = logging.getLogger("ray.rllib.agents.maml")
    logger.setLevel(logging.DEBUG)

    ray.init()
    env_cls = PointEnv
    register_env(env_cls.__name__,
                 lambda env_config: ResetWrapper(env_cls(), env_config))
    # register_env("PointEnv", lambda env_config: PointEnv(env_config))
    ModelCatalog.register_custom_model("maml_mlp", RLlibMLP)

    config = {
        "num_workers": 1,
        "model": {
            "custom_model": "maml_mlp",
            "fcnet_hiddens": [100, 100],
            "fcnet_activation": "tanh",
            "custom_options": {"vf_share_layers": True},
            # "squash_to_range": True,
            # "free_log_std": True
        }
    }

    agent = MAESNAgent(config=config, env=env_cls.__name__)
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

    writer = tf.summary.FileWriter(logdir="./summary", graph=evaluator.tf_sess.graph)
    writer.flush()
