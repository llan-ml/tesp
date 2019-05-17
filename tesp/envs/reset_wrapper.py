# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import ray
# from ray.experimental import named_actors


class ResetWrapper(gym.Wrapper):
    def __init__(self, env, env_config):
        assert not isinstance(env, self.__class__)
        gym.Wrapper.__init__(self, env)
        self.env_config = env_config
        self.reset_args_holder = self.env_config.get("reset_args_holder")
        # set the following attribute in MAMLPolicyEvaluator.reset_sample
        self.with_reset_args = None

    @property
    def reset_args_config(self):
        return self.env.reset_args_config

    def sample_reset_args(self, rng, num_train,
                          num_test_1=None, num_test_2=None):
        num_test_1 = num_test_1 or num_train
        num_test_2 = num_test_2 or num_train
        ret_train, ret_test_1, ret_test_2 = \
            self._sample_reset_args_near_and_far(
                rng, num_train, num_test_1, num_test_2)
        ret_train = np.stack(ret_train)
        ret_test_1 = np.stack(ret_test_1)
        ret_test_2 = np.stack(ret_test_2)
        return ret_train, ret_test_1, ret_test_2

    def _sample_reset_args_near_and_far(
            self, rng, num_train, num_test_1, num_test_2):
        low = self.reset_args_config["low"]
        high = self.reset_args_config["high"]
        threshold = self.reset_args_config["threshold"]
        sample_threshold = self.reset_args_config["sample_threshold"]
        sample_func = self.env.sample_reset_args_func(rng, low, high)
        ret_train = []
        ret_test_1 = []
        ret_test_2 = []
        while len(ret_train) < num_train:
            tmp = sample_func()
            if 0.2 < np.linalg.norm(tmp) < threshold:
                if not any([np.linalg.norm(tmp - x) < sample_threshold
                            for x in ret_train]):
                    ret_train.append(tmp)
        while len(ret_test_1) < num_test_1:
            tmp = sample_func()
            if 0.2 < np.linalg.norm(tmp) < threshold:
                if not any([np.linalg.norm(tmp - x) < sample_threshold
                            for x in ret_test_1 + ret_train]):
                    ret_test_1.append(tmp)
        while len(ret_test_2) < num_test_2:
            tmp = sample_func()
            if threshold < np.linalg.norm(tmp) < high:
                if not any([np.linalg.norm(tmp - x) < sample_threshold
                            for x in ret_test_2]):
                    ret_test_2.append(tmp)
        return ret_train, ret_test_1, ret_test_2

    def _sample_reset_args_left_and_right(
            self, rng, num_train, num_test_1, num_test_2):
        left_low = [-2.0, -2.0]
        left_high = [0.0, 2.0]
        right_low = [0.0, -2.0]
        right_high = [2.0, 2.0]
        left_sample_func = self.env.sample_reset_args_func(
            rng, left_low, left_high)
        right_sample_func = self.env.sample_reset_args_func(
            rng, right_low, right_high)
        ret_train = []
        ret_test_1 = []
        ret_test_2 = []
        while len(ret_train) < num_train:
            tmp = right_sample_func()
            if not any([np.allclose(tmp, x, atol=0.01) for x in ret_train]):
                ret_train.append(tmp)
        while len(ret_test_1) < num_test_1:
            tmp = right_sample_func()
            if not any([np.allclose(tmp, x, atol=0.01)
                        for x in ret_train + ret_test_1]):
                ret_test_1.append(tmp)
        while len(ret_test_2) < num_test_2:
            tmp = left_sample_func()
            if not any([np.allclose(tmp, x, atol=0.01) for x in ret_test_2]):
                ret_test_2.append(tmp)
        return ret_train, ret_test_1, ret_test_2

    def reset(self):
        # reset_args = ray.get(
        #     named_actors.get_actor("reset_args").get.remote())
        if self.with_reset_args:
            this_reset_args = self.reset_args
        else:
            # reset_args = ray.get(self.reset_args_holder.get.remote())
            # this_reset_args = reset_args[self.env_config.worker_index - 1]
            this_reset_args = ray.get(
                self.reset_args_holder.get_at.remote(
                    self.env_config.worker_index - 1))
            self.reset_args = this_reset_args
            self.with_reset_args = True
        return self.env.reset(this_reset_args)

    def step(self, action):
        return self.env.step(action)


@ray.remote(num_cpus=1)
class ResetArgsHolder(object):
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.args = np.zeros(shape)

    def get(self):
        return self.args

    def set(self, args):
        assert args.shape == self.shape
        self.args = args

    def get_at(self, index):
        return self.args[index]

    # def set_at(self, index, args):
    #     self.args[index] = args
