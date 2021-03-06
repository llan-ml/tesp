# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from envs.mujoco.mujoco_env import MujocoEnv
from envs.mujoco.utils import get_asset_xml


class ReacherEnv(MujocoEnv):
    xml_name = "reacher.xml"
    frame_skip = 5

    def __init__(self, env_config=None):
        self._env_config = env_config or {}
        self._ctrl_cost_coeff = self._env_config.get("ctrl_cost_coeff", 0.0)
        self._seed = self._env_config.get("seed", 123)
        self._goal = np.array([0.71, 0.71])
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

        self.do_simulation(action, self.frame_skip)
        vec = self.get_body_com("fingertip")[:2] - self._goal
        dist_reward = - np.linalg.norm(vec)
        if self._init:
            ctrl_cost = self.calculate_ctrl_cost(action)
        else:
            ctrl_cost = 0.0
        reward = dist_reward - self._ctrl_cost_coeff * ctrl_cost
        done = np.linalg.norm(vec) < 0.06
        # done = abs(vec[0]) < 0.05 and abs(vec[1]) < 0.05
        obs = self._get_obs()
        return obs, reward, done, {"goal": self._goal,
                                   "dist_reward": dist_reward}

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qpos.flat[2:],
            self.data.qvel.flat[:2],
            self.get_body_com("fingertip")[:2]
        ])

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        return self._get_obs()
