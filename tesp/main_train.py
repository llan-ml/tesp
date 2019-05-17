# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ray
from ray import tune
from ray.tune.registry import register_trainable, register_env
from ray.tune import run_experiments, grid_search
from ray.tune.trial import date_str
from ray.rllib.models.catalog import ModelCatalog

from maml.maml import MAMLAgent
from maml.meta_sgd import MetaSGDAgent
from maml.maesn import MAESNAgent
from tesp.tesp import TESPAgent

from envs.reset_wrapper import ResetWrapper
from rllib_models.rllib_mlp import RLlibMLP
from rllib_models.rllib_maesn import RLlibMAESN
from rllib_models.rllib_tesp import RLlibTESP
from rllib_models.rllib_tesp_with_adap_policy import RLlibTESPWithAdapPolicy


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="wheeled",
                    choices=["wheeled",
                             "ant",
                             "reacher_2-link",
                             "reacher_4-link"])
parser.add_argument("--alg", default="tesp",
                    choices=["maml",
                             "meta-sgd",
                             "maesn",
                             "tesp"])
parser.add_argument("--tesp_adapt_policy", action="store_true", default=False)
args = parser.parse_args()

env_name = args.env
alg_name = args.alg

# ray.init()
ray.init(redis_address="192.168.12.190:32222")

if env_name == "wheeled":
    from envs.mujoco.wheeled import WheeledEnv
    env_cls = WheeledEnv
elif env_name == "ant":
    from envs.mujoco.ant import AntEnv
    env_cls = AntEnv
elif env_name == "reacher_2-link":
    from envs.mujoco.reacher import ReacherEnv
    env_cls = ReacherEnv
elif env_name == "reacher_4-link":
    from envs.mujoco.reacher_4_link import Reacher4LinkEnv
    env_cls = Reacher4LinkEnv

if alg_name == "maml":
    agent_cls = MAMLAgent
elif alg_name == "meta-sgd":
    agent_cls = MetaSGDAgent
elif alg_name == "maesn":
    agent_cls = MAESNAgent
elif alg_name == "tesp":
    agent_cls = TESPAgent

all_agent_cls = [agent_cls]
all_model_cls = [RLlibMLP, RLlibMAESN, RLlibTESP, RLlibTESPWithAdapPolicy]

register_env(env_cls.__name__,
             lambda env_config: ResetWrapper(env_cls(env_config), env_config))
for agent_cls in all_agent_cls:
    register_trainable(agent_cls._agent_name, agent_cls)
for model_cls in all_model_cls:
    ModelCatalog.register_custom_model(model_cls.__name__, model_cls)


def get_config(model_cls_name):
    config = {
        "random_seed": grid_search([1]),
        "inner_lr": grid_search([0.001]),
        "inner_lr_bound": 0.1,
        "inner_grad_clip": 60.0,
        "num_inner_updates": 3,
        "outer_lr": grid_search([3e-4]),
        "num_sgd_iter": 20,
        "clip_param": 0.15,
        "model_loss_coeff": grid_search([0.01]),
        "validation": True,
        "validation_freq": 25,
        "num_cpus_per_worker": 1,
        "gamma": 0.99,
        "lambda": 0.97,
        "env_config": {
            "ctrl_cost_coeff": grid_search([1e-3]),
            "seed": lambda spec: int(spec.config.random_seed)
        },
        "model": {
            "custom_model": model_cls_name,
            "custom_options": {
                "rnn_units": grid_search([256]),
                "rnn_output_units": grid_search([8]),
                "mlp_hidden_units": grid_search([[256, 256]]),
                "projection_kernel_init_value": 0.01,
                "vf_share_layers": False,
                "linear_baseline": True}
        }
    }
    return config


experiment_specs = {}
for idx, agent_cls in enumerate(all_agent_cls):
    experiment_name = \
        f"TRAIN_{agent_cls._agent_name}_{env_cls.__name__}_{date_str()}"
    if agent_cls._agent_name == TESPAgent._agent_name:
        if not args.tesp_adapt_policy:
            config = get_config(RLlibTESP.__name__)
        else:
            config = get_config(RLlibTESPWithAdapPolicy.__name__)
            experiment_name = \
                f"TRAIN_{agent_cls._agent_name}AdaptPolicy" \
                f"_{env_cls.__name__}_{date_str()}"
    else:
        config = get_config(RLlibMLP.__name__)

    if agent_cls._agent_name in \
            [MAMLAgent._agent_name, MetaSGDAgent._agent_name]:
        config["model_loss_coeff"] = 0.0

    experiment_specs.update(
        {
            experiment_name: {
                "run": agent_cls._agent_name,
                "env": env_cls.__name__,
                "stop": {"training_iteration": 2},
                "config": config,
                "local_dir": "/tmp/ray_results",
                "max_failures": 0,
            }
        }
    )

run_experiments(experiment_specs)
