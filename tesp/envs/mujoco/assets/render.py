# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import mujoco_py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-n", default="reacher_4-link")

args = parser.parse_args()
xml_name = args.n

model = mujoco_py.load_model_from_path(f"./{xml_name}.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# for _ in range(100):
#     sim.data.ctrl[:] = np.random.uniform(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
#     sim.step()

while True:
    sim.data.ctrl[:] = np.random.uniform(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    for _ in range(5):
        sim.step()
    viewer.render()
