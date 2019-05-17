#!/bin/bash

set -e

source env.sh

echo "starting on ${head_node}"
ssh -t ${head_node} " /bin/bash -i -c '
  source \${HOME}/.bashrc
  # echo \${PATH}
  export PYTHONPATH=${PYTHONPATH}
  ulimit -n
  ulimit -n 65536
  ulimit -n
  echo \$(which ray)
  ray start --head --node-ip-address ${head_node_ip} --redis-port ${port} --object-store-memory ${plasma_size} --num-redis-shards ${redis_shards} --redis-max-memory ${redis_memory_per_shard}
  sleep 2
'"

host_string=""
for node in ${nodes}
do
  host_string+="${node} "
done

parallel-ssh -i -H "${host_string}" -x "-t -t" " /bin/bash -i -c '
  source \${HOME}/.bashrc
  # echo \${PATH}
  export PYTHONPATH=${PYTHONPATH}
  # echo \${SSH_CLIENT}
  # echo \${SSH_CONNECTION}
  ulimit -n
  ulimit -n 65536
  ulimit -n
  echo \$(which ray)
  node_ip_address=\$(echo \${SSH_CONNECTION} | awk "'"{print \$3}"'" | awk -F "." "'"{print \$4}"'")
  # echo \${node_ip_address}
  ray start --node-ip-address 192.168.12.\${node_ip_address} --redis-address ${head_node_ip}:${port} --object-store-memory ${plasma_size}
  sleep 0.5
'"
