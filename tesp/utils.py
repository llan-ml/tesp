# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import defaultdict
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import discount, compute_advantages
from ray.rllib.evaluation.metrics \
    import summarize_episodes as _summarize_episodes


def compute_returns(rollout, last_r, gamma):
    traj = {}
    trajsize = len(rollout["actions"])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    rewards_plus_v = np.concatenate(
        [rollout["rewards"], np.array([last_r])])
    traj["returns"] = discount(rewards_plus_v, gamma)[:-1]

    traj["returns"] = traj["returns"].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


def separate_sample_batch(sample_batch):
    separated_sample_batch = defaultdict(lambda: defaultdict(list))
    for i, eps_id in enumerate(sample_batch["eps_id"]):
        for key in sample_batch.keys():
            separated_sample_batch[eps_id][key].append(sample_batch[key][i])
    for eps_id, values in separated_sample_batch.items():
        for k, v in values.items():
            separated_sample_batch[eps_id][k] = np.stack(v)
        separated_sample_batch[eps_id] = SampleBatch(
            dict(separated_sample_batch[eps_id]))
    separated_sample_batch = dict(separated_sample_batch)
    return separated_sample_batch


def postprocess_trajectory(samples, baseline, gamma, lambda_, use_gae):
    separated_samples = separate_sample_batch(samples)
    baseline.fit(separated_samples.values())
    for eps_id, values in separated_samples.items():
        values["vf_preds"] = baseline.predict(values)
        separated_samples[eps_id] = compute_advantages(
            values,
            0.0,
            gamma,
            lambda_,
            use_gae)
    samples = SampleBatch.concat_samples(
        list(separated_samples.values()))
    return samples


def summarize_episodes(episodes, new_episodes, num_dropped):
    metrics = _summarize_episodes(episodes, new_episodes, num_dropped)
    for key in ["episode_reward_max", "episode_reward_min",
                "policy_reward_mean",
                "num_metric_batches_dropped"]:
        del metrics[key]
    return metrics
