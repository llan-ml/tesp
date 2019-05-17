# Meta Reinforcement Learning

This repository contains the TensorFlow implementation of our paper
[Meta Reinforcement Learning with Task Embedding and Shared Policy](https://arxiv.org/abs/1905.06527) (IJCAI 2019).

For detailed settings of the environments and experiments,
please refer to the [supplementary material](supplement.pdf).

We also re-implement the following methods:

- MAML for RL
    - [Agent Class](tesp/maml/maml.py)
    - [paper](https://arxiv.org/abs/1703.03400)
- Meta-SGD for RL
    - [Agent Class](tesp/maml/meta_sgd.py)
    - [paper](https://arxiv.org/abs/1707.09835)
- MAESN
    - [Agent Class](tesp/maml/maesn.py)
    - [paper](https://arxiv.org/abs/1802.07245)
    - Note that here we use a latent variable instead of a latent distribution.

## Requirements

- `python==3.6.5`
- `tensorflow>=1.11,<2.0`
- `ray>=0.6`
(we used this [version](https://github.com/ray-project/ray/commits/5dcc33319919c9c4581b823a87452527b29c50a3) in the experiments,
 but any version `ray>=0.6` should work)
- `gym>=0.10`
- `mujoco-py==1.50.1.68`

We recommend installing [Anaconda](https://repo.anaconda.com/archive/) before installing other dependencies.

## Usage

### Preparation

We provide several samples of bash scripts to ease operations on a Ray cluster:

- `env.sh`: Declare the configuration of Ray cluster.
- `update_ray_codes.sh`: Replace with our slightly modified [RLlib](ray/rllib) and [Tune](ray/tune).
- `exec_commands.sh`: Make some directories and (optionally) install ray on each node.
- `ray_start.sh`: Start a Ray cluster.
- `ray_stop.sh`: Stop a Ray cluster.
- `upload_codes.sh`: Upload the training code to each node of a Ray cluster.

### Run

#### Distributed Mode

You need to first launch a Ray cluster declared in `env.sh` and upload relevant codes to each node.

Then, log into the master (head) node, switch to the working directory (see `upload_codes.sh`), and type

```bash
python main_train.py --env wheeled --alg tesp
```

#### Local Mode

You need to uncomment `ray.init()` in [main_train.py](tesp/main_train.py).

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{lan2019metarl,
  title={Meta Reinforcement Learning with Task Embedding and Shared Policy},
  author={Lan, Lin and Li, Zhenguo and Guan, Xiaohong and Wang, Pinghui},
  booktitle={IJCAI},
  year={2019}
}
```
