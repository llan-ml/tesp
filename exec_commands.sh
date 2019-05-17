#!/bin/bash

source env.sh

ray_version="5dcc33319919c9c4581b823a87452527b29c50a3"

# parallel-scp -t 0 -H "${servers}" -r ray_versions/${ray_version} /tmp

parallel-ssh -i -H "${servers}" -x "-t -t" " /bin/bash -i -c '
  source \${HOME}/.bashrc
  mkdir -p \${HOME}/Workspaces/tesp
  mkdir -p /tmp/ray_results
  rm -r \${HOME}/Workspaces/tesp/*
  rm -r /tmp/ray/*
  rm -r /tmp/ray_results/*
  rm \${HOME}/mujoco-py/mujoco_py/generated/*lock
  python -c \"import mujoco_py; print(\\\"mujoco_py pass\\\")\"
  # pip uninstall -y ray
  # pip install /tmp/${ray_version}/ray-0.6.0-cp36-cp36m-manylinux1_x86_64.whl
  # pip install ray[debug]
  echo done
'"
