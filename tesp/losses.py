# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ray.rllib.models.catalog import ModelCatalog


class A3CLoss(object):
    def __init__(self,
                 action_dist,
                 actions,
                 advantages,
                 value_targets,
                 vf_preds,
                 value_function,
                 vf_loss_coeff=1.0,
                 entropy_coeff=0.0,
                 vf_clip_param=None):
        log_prob = action_dist.logp(actions)

        self.mean_policy_loss = - tf.reduce_mean(log_prob * advantages)

        vf_loss = get_vf_loss(value_function, value_targets,
                              vf_preds, vf_clip_param)
        self.mean_vf_loss = tf.reduce_mean(vf_loss)

        if entropy_coeff:
            self.mean_entropy = tf.reduce_mean(action_dist.entropy())
        else:
            self.mean_entropy = tf.constant(0.0)

        self.loss = self.mean_policy_loss + vf_loss_coeff * self.mean_vf_loss \
            - entropy_coeff * self.mean_entropy


class PPOLoss(object):
    def __init__(self,
                 action_dist,
                 action_space,
                 logits,
                 actions,
                 advantages,
                 value_targets,
                 vf_preds,
                 value_function,
                 valid_mask,
                 kl_coeff,
                 clip_param,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 entropy_coeff=0.0,
                 use_gae=True):
        dist_cls, _ = ModelCatalog.get_action_dist(action_space, {})
        old_dist = dist_cls(logits)

        logp_ratio = tf.exp(
            action_dist.logp(actions) - old_dist.logp(actions))
        action_kl = old_dist.kl(action_dist)
        self.mean_kl = tf.reduce_mean(action_kl)

        if kl_coeff:
            self.kl_coeff = tf.get_variable(
                initializer=tf.constant_initializer(kl_coeff),
                name="kl_coeff",
                shape=(),
                trainable=False,
                dtype=tf.float32)
        else:
            self.kl_coeff = tf.constant(0.0)

        if entropy_coeff:
            entropy = action_dist.entropy()
            self.mean_entropy = tf.reduce_mean(entropy)
        else:
            self.mean_entropy = tf.constant(0.0)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio,
                                          1 - clip_param, 1 + clip_param))
        self.mean_policy_loss = - tf.reduce_mean(surrogate_loss)

        if use_gae:
            vf_loss = get_vf_loss(value_function, value_targets,
                                  vf_preds, vf_clip_param)
            self.mean_vf_loss = tf.reduce_mean(vf_loss)
        else:
            raise NotImplementedError

        self.loss = self.mean_policy_loss + self.kl_coeff * self.mean_kl \
            + vf_loss_coeff * self.mean_vf_loss \
            - entropy_coeff * self.mean_entropy


def get_vf_loss(value_function, value_targets,
                vf_preds=None, vf_clip_param=None):
    if value_function is not None:
        if vf_clip_param:
            vf_loss_1 = tf.square(value_function - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_function - vf_preds, - vf_clip_param, vf_clip_param)
            vf_loss_2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss_1, vf_loss_2)
        else:
            vf_loss = tf.square(value_function - value_targets)
    else:
        vf_loss = tf.constant(0.0, dtype=tf.float32)
    return vf_loss
