# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from envs.mujoco.mujoco_env import MujocoEnv
from envs.mujoco.utils import get_asset_xml


class WheeledEnv(MujocoEnv):
    xml_name = "wheeled.xml"
    frame_skip = 5

    def __init__(self, env_config=None):
        self._env_config = env_config or {}
        self._ctrl_cost_coeff = self._env_config.get("ctrl_cost_coeff", 0.0)
        self._seed = self._env_config.get("seed", 123)
        self._goal = np.array([1.4142, 1.4142])
        self._rng = np.random.RandomState(self._seed)
        xml_path = get_asset_xml(self.xml_name)
        MujocoEnv.__init__(self, xml_path, self.frame_skip)

    @property
    def reset_args_config(self):
        return {
            "shape": (2, ),
            "low": -1.5,
            "high": 1.5,
            "threshold": 1.1,
            "sample_threshold": 0.1
        }

    def step(self, action):
        if self._init:
            action = self.normalize_action(action)

        # posbefore = self.get_body_com("car")[:2]
        self.do_simulation(action, self.frame_skip)
        posafter = self.get_body_com("car").copy()[:2]
        vec = posafter - self._goal
        dist_reward = - np.linalg.norm(vec)
        if self._init:
            ctrl_cost = self.calculate_ctrl_cost(action)
        else:
            ctrl_cost = 0.0
        reward = dist_reward - self._ctrl_cost_coeff * ctrl_cost
        done = np.linalg.norm(vec) < 0.02
        # done = abs(x) < 0.02 and abs(y) < 0.02
        obs = self._get_obs()
        return obs, reward, done, {"goal": self._goal,
                                   "dist_reward": dist_reward,
                                   "ctrl_cost": ctrl_cost}

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        ori = [np.sin(qpos[-3]), np.cos(qpos[-3])]
        qvel = self.data.qvel.flatten()
        obs = np.concatenate([qpos[:-3], ori, qvel[:-2]])
        return obs

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        # qpos[-2:] = self._goal.copy()
        self.set_state(qpos, qvel)
        return self._get_obs()
