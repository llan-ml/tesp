#!/bin/bash

set -e

export port="32222"
export head_node_ip="192.168.12.190"
export head_node="server-190"
export nodes="server-191"
export plasma_size="20000000000"
export redis_shards="1"
export redis_memory_per_shard="100000000000"
export PYTHONPATH="\${HOME}/Workspaces/tesp"

export servers="server-190 server-191"
