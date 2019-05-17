# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import init_ops


class NormcInitializer(init_ops.Initializer):
    def __init__(self, std, dtype=tf.float32):
        self.std = std
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        randn = tf.random_normal(shape=shape, dtype=dtype)
        out = randn * self.std / tf.sqrt(
            tf.reduce_sum(tf.square(randn),
                          axis=0, keepdims=True))
        return out

    def get_config(self):
        return {
            "std": self.std,
            "dtype": self.dtype.name
        }


normc_initializer = NormcInitializer
