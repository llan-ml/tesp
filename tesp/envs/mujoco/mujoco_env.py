# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym.envs.mujoco import mujoco_env


class MujocoEnv(mujoco_env.MujocoEnv):
    def __init__(self, model_path, frame_skip):
        self._init = False
        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)
        self._init = True

    def reset(self, reset_args=None):
        assert isinstance(reset_args, np.ndarray) and reset_args.ndim == 1
        reset_args = np.copy(reset_args)
        goal = reset_args
        self._goal = goal
        return mujoco_env.MujocoEnv.reset(self)

    def corrupt_obs(self, obs):
        corrupted_obs = np.copy(obs)
        if self._mode == 1:
            noise = self._rng.normal(0.0, self._sigma, size=obs.shape)
            corrupted_obs += noise
        return corrupted_obs

    def _cripple_action(self, action):
        cripple_action = np.copy(action)
        # if self._mode == 1:
        assert self._to_cripple in list(range(self.action_space.shape[0]))
        cripple_action[self._to_cripple] = 0.0
        return cripple_action

    def cripple_action(self, action):
        crippled_action = np.copy(action)
        if self._mode == 1:
            # if np.random.random() > (1 - self._cripple_rate):
            to_cripple = self._rng.choice(self.action_space.shape[0])
            assert to_cripple in list(range(self.action_space.shape[0]))
            crippled_action[to_cripple] = 0.0
        return crippled_action

    def normalize_action(self, action):
        lb = self.action_space.low
        ub = self.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

    def calculate_ctrl_cost(self, action):
        lb = self.action_space.low
        ub = self.action_space.high
        scaling = (ub - lb) * 0.5
        return 0.5 * np.sum(np.square(action / scaling))

    def sample_reset_args_func(self, rng, low, high):
        return lambda: rng.uniform(low=low, high=high,
                                   size=self.reset_args_config["shape"])
